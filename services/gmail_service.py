# services/gmail_service.py

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import base64
import datetime


from core.constants.gmail_constants import CREDENTIALS_FILE, SCOPES
from core.utils.gmail_utils import get_credentials, save_credentials_to_firestore


router = APIRouter()


@router.get("/login")
def login():
    flow = Flow.from_client_secrets_file(
        CREDENTIALS_FILE,
        scopes=SCOPES,
        redirect_uri="http://127.0.0.1:8000/gmail/oauth2callback"
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )
    return RedirectResponse(auth_url)


@router.get("/oauth2callback")
def oauth2callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        return {"error": "Missing code parameter"}

    flow = Flow.from_client_secrets_file(
        CREDENTIALS_FILE,
        scopes=SCOPES,
        redirect_uri="http://127.0.0.1:8000/gmail/oauth2callback"
    )
    flow.fetch_token(code=code)
    creds = flow.credentials

    service = build("gmail", "v1", credentials=creds)
    profile = service.users().getProfile(userId="me").execute()
    user_id = profile["emailAddress"]

    save_credentials_to_firestore(user_id, creds)

    return {"message": f"Authentication successful, credentials saved for {user_id}"}

@router.get("/read-emails/")
def read_emails(user_id: str = None, max_results: int = 5):
    """
    Read emails either for a single user (if user_id provided) 
    or for all users stored in Firestore.
    """
    email_results = {}

    try:
        if user_id:
            creds = get_credentials(user_id)
            service = build("gmail", "v1", credentials=creds)
            email_results[user_id] = fetch_emails(service, user_id, max_results)
        else:
            from core.utils.firebase_utils import db
            docs = db.collection("gmail_tokens").stream()
            for doc in docs:
                uid = doc.id
                creds = get_credentials(uid)
                service = build("gmail", "v1", credentials=creds)
                email_results[uid] = fetch_emails(service, uid, max_results)

        return {"emails": email_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_emails(service, max_results: int = 5):
    """Helper to fetch and parse emails for a single Gmail account."""
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])
    email_list = []

    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()

        headers = msg_data["payload"].get("headers", [])
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), None)
        from_email = next((h["value"] for h in headers if h["name"] == "From"), None)

        body = ""
        if "parts" in msg_data["payload"]:
            for part in msg_data["payload"]["parts"]:
                if part["mimeType"] in ["text/plain", "text/html"] and "data" in part["body"]:
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                    break
        else:
            if "data" in msg_data["payload"]["body"]:
                body = base64.urlsafe_b64decode(msg_data["payload"]["body"]["data"]).decode("utf-8")

        attachments = []
        if "parts" in msg_data["payload"]:
            for part in msg_data["payload"]["parts"]:
                if part.get("filename") and "attachmentId" in part["body"]:
                    attachments.append({
                        "filename": part["filename"],
                        "mimeType": part["mimeType"],
                        "attachmentId": part["body"]["attachmentId"],
                        "download_url": f"/gmail/download-attachment/{msg['id']}/{part['body']['attachmentId']}"
                    })

        email_list.append({
            "id": msg["id"],
            "from": from_email,
            "subject": subject,
            "body": body,
            "attachments": attachments
        })

    return email_list

@router.get("/read-emails-history/")
def read_emails_history(user_id: str = None, days: int = 30, max_results: int = 100):
    """
    Fetch emails within the last `days` (default 30 days).
    Supports both single-user (if user_id is provided) and multi-user mode.
    """
    email_results = {}

    try:
        if user_id:
            creds = get_credentials(user_id)
            service = build("gmail", "v1", credentials=creds)
            email_results[user_id] = fetch_emails_in_date_range(service, days, max_results)
        else:
            from core.utils.firebase_utils import db
            docs = db.collection("gmail_tokens").stream()
            for doc in docs:
                uid = doc.id
                try:
                    creds = get_credentials(uid)
                    service = build("gmail", "v1", credentials=creds)
                    email_results[uid] = fetch_emails_in_date_range(service, days, max_results)
                except HTTPException as e:
                    email_results[uid] = {"error": e.detail}

        return {"emails": email_results}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import datetime

def fetch_emails_in_date_range(service, days: int = 30, max_results: int = 100):
    """
    Fetch emails from Gmail within the past `days`.
    """
    # Compute date N days ago
    after_date = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).date()
    after_query = f"after:{after_date.strftime('%Y/%m/%d')}"

    results = service.users().messages().list(
        userId="me",
        maxResults=max_results,
        q=after_query
    ).execute()

    messages = results.get("messages", [])
    email_list = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me", id=msg["id"], format="full"
        ).execute()

        headers = msg_data["payload"].get("headers", [])
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), None)
        from_email = next((h["value"] for h in headers if h["name"] == "From"), None)

        snippet = msg_data.get("snippet", "")
        email_list.append({
            "id": msg["id"],
            "from": from_email,
            "subject": subject,
            "snippet": snippet
        })

    return email_list
