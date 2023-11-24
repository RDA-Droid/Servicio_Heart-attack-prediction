from flask import Flask, request, jsonify, abort
from waitress import serve
from functools import wraps
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import uuid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# Configura tus claves API y proyectname
APIKEY = "db92efc69991"
PROYECTNAME = "demo"

# Middleware para verificar la clave API y el 'proyectname'
def check_credentials(func):
    @wraps(func)
    def check_credentials_wrapper(*args, **kwargs):
        api_key = request.headers.get("APIKEY")
        proyectname = request.headers.get("PROYECTNAME")

        if api_key is None or api_key != APIKEY:
            abort(401, "Acceso no autorizado. Proporcione una clave API válida.")

        if proyectname is None or proyectname != PROYECTNAME:
            abort(401, "Acceso no autorizado. Proporcione un proyecname válido.")

        return func(*args, **kwargs)

    return check_credentials_wrapper

# Initialize Firebase
cred = credentials.Certificate('D:/Roger/Desktop/temporales parcial/Servicio_Heart-attack-prediction/firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Generar un ID único para cada transacción
def generate_transaction_id():
    return str(uuid.uuid4())

# Load data from CSV file
archivo_csv = r'D:\Roger\Desktop\temporales parcial\Servicio_Heart-attack-prediction\datos.csv'
data = pd.read_csv(archivo_csv)

# Drop unnecessary columns
columns_to_drop = ['Patient ID']
data = data.drop(columns=columns_to_drop, errors='ignore')

# Check if 'Blood Pressure' is present in the columns
if 'Blood Pressure' in data.columns:
    # Split 'Blood Pressure' into 'Systolic Pressure' and 'Diastolic Pressure'
    data[['Systolic Pressure', 'Diastolic Pressure']] = data['Blood Pressure'].str.split('/', expand=True)

    # Convert new columns to numeric type
    data['Systolic Pressure'] = pd.to_numeric(data['Systolic Pressure'], errors='coerce')
    data['Diastolic Pressure'] = pd.to_numeric(data['Diastolic Pressure'], errors='coerce')

    # Drop the original 'Blood Pressure' column
    data = data.drop('Blood Pressure', axis=1)
else:
    print("La columna 'Blood Pressure' no está presente en el conjunto de datos.")

# Handle missing values
data = data.dropna()

# Convert categorical variables to one-hot encoding
data = pd.get_dummies(data)

# Separate features (X) and target variable (y)
X = data.drop('Heart Attack Risk', axis=1)
y = data['Heart Attack Risk']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Initialize the model and optimizer
input_size = X_train.shape[1]
model = SimpleNN(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))
    loss.backward()
    optimizer.step()

# Función de entrenamiento incremental
def incremental_train(model, criterion, optimizer, new_X, new_y):
    model.train()
    optimizer.zero_grad()
    outputs = model(new_X)
    loss = criterion(outputs, new_y.view(-1, 1))
    loss.backward()
    optimizer.step()

# Endpoint `/predict` modificado
@app.route('/predict/', methods=['POST'])
@check_credentials
def predict():
    try:
        # Receive the form data as JSON
        form_data = request.json

        # Ensure that form_data is a list of dictionaries
        if not isinstance(form_data, list):
            form_data = [form_data]

        # Create a DataFrame from the received data
        formulario_df = pd.DataFrame(form_data)

        # Preprocess the form data
        if 'Blood Pressure' in formulario_df.columns:
            formulario_df['Blood Pressure'] = formulario_df['Blood Pressure'].astype(str)
            split_result = formulario_df['Blood Pressure'].str.split('/', expand=True)
            if len(split_result.columns) == 1:
                formulario_df['Systolic Pressure'] = pd.to_numeric(split_result[0], errors='coerce')
                formulario_df['Diastolic Pressure'] = float('nan')
            else:
                formulario_df[['Systolic Pressure', 'Diastolic Pressure']] = split_result
                formulario_df['Systolic Pressure'] = pd.to_numeric(formulario_df['Systolic Pressure'], errors='coerce')
                formulario_df['Diastolic Pressure'] = pd.to_numeric(formulario_df['Diastolic Pressure'], errors='coerce')
            formulario_df = formulario_df.drop('Blood Pressure', axis=1)

        formulario_df = formulario_df.drop(columns=columns_to_drop, errors='ignore')
        formulario_df = pd.get_dummies(formulario_df)
        feature_names = X.columns
        formulario_df = formulario_df.reindex(columns=feature_names, fill_value=0)
        formulario_df = scaler.transform(formulario_df)
        formulario_tensor = torch.tensor(formulario_df, dtype=torch.float32)

        # Make a prediction
        model.eval()
        with torch.no_grad():
            prediction = model(formulario_tensor)
            probability = prediction.item()

        # Generate a unique ID for the transaction
        transaction_id = generate_transaction_id()

        # Insert results into Firebase
        db.collection('Predictions').add({
            'TransactionID': transaction_id,
            'Probability': probability,
            'OtherInfo': 'RESULTADO'
        })

        # Entrenar el modelo con los nuevos datos
        new_X = torch.tensor(formulario_df, dtype=torch.float32)
        new_y = torch.tensor([1.0 if probability >= 0.5 else 0.0], dtype=torch.float32)  # Etiqueta basada en la probabilidad

        incremental_train(model, criterion, optimizer, new_X, new_y)

        # Return the transaction ID in the response
        return jsonify({'transaction_id': transaction_id})

    except Exception as e:
        return jsonify({'error': str(e)})

# Endpoint `/obtener_prediccion/<transaction_id>` modificado
@app.route('/obtener_prediccion/<transaction_id>', methods=['GET'])
@check_credentials
def obtener_prediccion_endpoint(transaction_id):
    try:
        # Consultar la información de la base de datos utilizando el transaction_id
        result = db.collection('Predictions').where('TransactionID', '==', transaction_id).get()

        if not result:
            return jsonify({'error': 'No se encontró información para el ID de transacción proporcionado.'}), 404

        data = result[0].to_dict()
        transaction_id = data['TransactionID']
        probability = data['Probability']
        other_info = data['OtherInfo']

        return jsonify({'transaction_id': transaction_id, 'probability': probability, 'other_info': other_info})

    except Exception as e:
        return jsonify({'error': str(e)})

# For deploying with Waitress server (install it using: pip install waitress)
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
