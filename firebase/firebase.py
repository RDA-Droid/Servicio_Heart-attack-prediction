import firebase_admin
from firebase_admin import credentials, firestore
import uuid

cred = credentials.Certificate('C:/Users/USER/Desktop/brayan-project/Servicio_Heart-attack-prediction/firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def generate_transaction_id():
    return str(uuid.uuid4())
