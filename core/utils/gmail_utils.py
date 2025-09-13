
from fastapi import HTTPException
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from core.constants.gmail_constants import SCOPES
from core.utils.firebase_utils import db
from google.auth.exceptions import RefreshError
import datetime



def save_credentials_to_firestore(user_id: str, creds: Credentials):
    data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }
    db.collection("gmail_tokens").document(user_id).set(data)


def get_credentials(user_id: str) -> Credentials:
    doc = db.collection("gmail_tokens").document(user_id).get()
    if not doc.exists:
        raise HTTPException(status_code=401, detail="User not authenticated. Visit /gmail/login")

    data = doc.to_dict()

    creds = Credentials(
        token=data.get("token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri"),
        client_id=data.get("client_id"),
        client_secret=data.get("client_secret"),
        scopes=data.get("scopes"),
    )

    if "expiry" in data:
        try:
            creds.expiry = datetime.datetime.fromisoformat(data["expiry"])
        except Exception:
            creds.expiry = None

    if creds.expired:
        if creds.refresh_token:
            try:
                creds.refresh(GoogleRequest())
                save_credentials_to_firestore(user_id, creds)
            except RefreshError as e:
                raise HTTPException(status_code=401, detail=f"Refresh token invalid: {str(e)}")
        else:
            raise HTTPException(status_code=401, detail="No refresh token available. Re-authenticate.")

    return creds

