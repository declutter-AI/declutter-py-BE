import os

CREDENTIALS_FILE = os.getenv("GOOGLE_OAUTH_CREDENTIALS", "./credentials.json")
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify"
]
