import firebase_admin
from firebase_admin import credentials, firestore, auth
import uuid
import json
import jwt
from datetime import datetime, timedelta

cred = credentials.Certificate('C:/Users/USER/Desktop/brayan-project/Servicio_Heart-attack-prediction/firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

SECRET_KEY = 'tu_clave_secreta'
USERS_FILE = 'C:/Users/USER/Desktop/brayan-project/Servicio_Heart-attack-prediction/users.json'

def generate_transaction_id():
    return str(uuid.uuid4())

def register_user(email, password):
    try:
        with open(USERS_FILE, 'r') as f:
            users_data = json.load(f)

        # Verificar si el usuario ya está registrado
        for user in users_data['users']:
            if user['email'] == email:
                raise Exception('El usuario ya está registrado.')

        # Agregar el nuevo usuario
        new_user = {'email': email, 'password': password}
        users_data['users'].append(new_user)

        with open(USERS_FILE, 'w') as f:
            json.dump(users_data, f, indent=2)

        return 'Usuario registrado exitosamente.'

    except Exception as e:
        return str(e)

def login_user(email, password):
    try:
        with open(USERS_FILE, 'r') as f:
            users_data = json.load(f)

        # Verificar las credenciales
        for user in users_data['users']:
            if user['email'] == email and user['password'] == password:
                # Generar un token JWT con información del usuario
                payload = {
                    'email': user['email'],
                    'exp': datetime.utcnow() + timedelta(days=1)  # Token válido por 1 día
                }
                token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')

                return {'message': 'Inicio de sesión exitoso.', 'token': token}

        raise Exception('Credenciales incorrectas.')

    except Exception as e:
        return {'error': str(e)}
