import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
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

    return X_train, X_test, y_train, y_test, scaler

