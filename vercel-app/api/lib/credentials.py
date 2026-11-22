"""
Credential helpers for Vercel serverless functions.
Reads DRIVE_TOKEN_JSON, DRIVE_INPUT_FOLDER_ID, DRIVE_OUTPUT_FOLDER_ID from env vars.
"""

import os
import json
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/drive"]

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

DRIVE_INPUT_FOLDER_ID = os.environ.get(
    "DRIVE_INPUT_FOLDER_ID", "1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88"
)
DRIVE_OUTPUT_FOLDER_ID = os.environ.get(
    "DRIVE_OUTPUT_FOLDER_ID", "1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88"
)


def get_drive_credentials() -> Credentials:
    """Build Google OAuth Credentials from the DRIVE_TOKEN_JSON env var."""
    raw = os.environ.get("DRIVE_TOKEN_JSON", "")
    if not raw:
        raise RuntimeError("DRIVE_TOKEN_JSON environment variable is not set")
    token_info = json.loads(raw)
    creds = Credentials.from_authorized_user_info(token_info, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            pass
    return creds
