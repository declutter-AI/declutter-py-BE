import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    firebase_config = os.getenv("FIREBASE_CONFIG")  
    firebase_path = os.getenv("FIREBASE_CRED_PATH") 

    if firebase_config:
        cred = credentials.Certificate(json.loads(firebase_config))
    elif firebase_path:
        cred = credentials.Certificate(os.path.expanduser(firebase_path))
    else:
        raise ValueError("No Firebase credentials provided!")

    firebase_admin.initialize_app(cred)

db = firestore.client()
