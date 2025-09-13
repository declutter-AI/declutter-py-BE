# core/utils/gmail_utils.py

from fastapi import HTTPException
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from core.constants.gmail_constants import SCOPES
from core.utils.firebase_utils import db


def save_credentials_to_firestore(user_id: str, creds: Credentials):
    """
    Save Gmail credentials for a user into Firestore.
    """
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }
    db.collection("gmail_tokens").document(user_id).set(token_data)


def get_credentials(user_id: str) -> Credentials:
    """
    Load Gmail credentials for a user from Firestore.
    """
    doc = db.collection("gmail_tokens").document(user_id).get()
    if not doc.exists:
        raise HTTPException(status_code=401, detail="User not authenticated. Visit /gmail/login")

    data = doc.to_dict()
    creds = Credentials.from_authorized_user_info(data, SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(GoogleRequest())
        save_credentials_to_firestore(user_id, creds)

    return creds
