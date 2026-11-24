"""
Stateless Google Drive helper for Vercel serverless functions.
Each function call creates its own service — no shared state.
"""

import time
import socket
import ssl
from typing import List, Dict, Optional, Callable
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from .credentials import get_drive_credentials, DRIVE_INPUT_FOLDER_ID, DRIVE_OUTPUT_FOLDER_ID


def _build_service():
    creds = get_drive_credentials()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _is_transient(error: Exception) -> bool:
    if isinstance(error, (socket.timeout, ConnectionError, ssl.SSLError)):
        return True
    s = str(error).lower()
    return any(
        k in s
        for k in [
            "decryption_failed",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "deadline exceeded",
            "bad status line",
        ]
    )


def _retry(fn, description: str, retries: int = 3, backoff: float = 1.5):
    last = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if not _is_transient(e) or attempt == retries:
                break
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"{description}: {last}")


def get_all_projects() -> List[Dict]:
    svc = _build_service()
    projects: List[Dict] = []
    page_token = None
    while True:
        params = {
            "q": f"'{DRIVE_INPUT_FOLDER_ID}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
            "fields": "nextPageToken, files(id, name)",
            "pageSize": 1000,
            "orderBy": "name",
        }
        if page_token:
            params["pageToken"] = page_token
        results = _retry(lambda p=params: svc.files().list(**p).execute(), "list projects")
        for f in results.get("files", []):
            fid = f.get("id")
            if fid:
                projects.append({"id": fid, "name": f.get("name") or f"Unnamed ({fid[:8]})"})
        page_token = results.get("nextPageToken")
        if not page_token:
            break
    seen = set()
    unique = []
    for p in projects:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)
    unique.sort(key=lambda x: x["name"].lower())
    return unique


def get_brandings(project_id: str) -> List[Dict]:
    svc = _build_service()
    brandings: List[Dict] = []
    page_token = None
    while True:
        params = {
            "q": f"'{project_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
            "fields": "nextPageToken, files(id, name)",
            "pageSize": 100,
        }
        if page_token:
            params["pageToken"] = page_token
        results = _retry(lambda p=params: svc.files().list(**p).execute(), "list brandings")
        for f in results.get("files", []):
            fid = f.get("id")
            if fid:
                brandings.append({"id": fid, "name": f.get("name") or f"Unnamed ({fid[:8]})"})
        page_token = results.get("nextPageToken")
        if not page_token:
            break
    brandings.sort(key=lambda x: x["name"].lower())
    return brandings


def get_file_types(branding_id: str) -> List[str]:
    svc = _build_service()
    try:
        raw = svc.files().list(
            q=f"name='Raw' and '{branding_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
            fields="files(id)",
            pageSize=1,
        ).execute()
        folders = raw.get("files", [])
        if not folders:
            return []
        raw_id = folders[0]["id"]
        types: List[str] = []
        page_token = None
        while True:
            params = {
                "q": f"'{raw_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
                "fields": "nextPageToken, files(name)",
                "pageSize": 100,
                "orderBy": "name",
            }
            if page_token:
                params["pageToken"] = page_token
            res = svc.files().list(**params).execute()
            types.extend(f["name"] for f in res.get("files", []))
            page_token = res.get("nextPageToken")
            if not page_token:
                break
        types.sort()
        return types
    except Exception:
        return []


def find_or_create_folder(svc, parent_id: str, name: str) -> str:
    res = _retry(
        lambda: svc.files().list(
            q=f"name='{name}' and '{parent_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
            fields="files(id)",
            pageSize=1,
        ).execute(),
        f"find folder '{name}'",
    )
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    folder = _retry(
        lambda: svc.files().create(body=meta, fields="id").execute(),
        f"create folder '{name}'",
    )
    return folder["id"]


def list_files_in_folder(svc, folder_id: str) -> List[Dict]:
    files: List[Dict] = []
    page_token = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false and mimeType!='application/vnd.google-apps.folder'",
            "fields": "nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            "pageSize": 1000,
        }
        if page_token:
            params["pageToken"] = page_token
        res = _retry(lambda p=params: svc.files().list(**p).execute(), "list files")
        files.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    files.sort(key=lambda x: (x.get("name") or "").lower())
    return files


def trash_files(svc, folder_id: str) -> List[str]:
    files = list_files_in_folder(svc, folder_id)
    deleted: List[str] = []
    for f in files:
        fid = f["id"]
        _retry(
            lambda fid=fid: svc.files().update(fileId=fid, body={"trashed": True}).execute(),
            f"trash '{f.get('name', fid)}'",
        )
        deleted.append(f.get("name", fid))
    return deleted


def resolve_destination(project_name: str, branding_name: str, file_type: str):
    svc = _build_service()
    project_folder = find_or_create_folder(svc, DRIVE_OUTPUT_FOLDER_ID, project_name)
    branding_folder = find_or_create_folder(svc, project_folder, branding_name)
    res = svc.files().list(
        q=f"'{branding_folder}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
        fields="files(id, name)",
        pageSize=10,
    ).execute()
    project_name_folder = None
    clean_project_name = None
    for folder in res.get("files", []):
        if folder["name"] not in ["Raw", "AI data"]:
            project_name_folder = folder
            clean_project_name = folder["name"]
            break
    if not project_name_folder:
        raise RuntimeError(f"No project name folder under {branding_name}")
    file_type_id = find_or_create_folder(svc, project_name_folder["id"], file_type)
    return {
        "svc": svc,
        "file_type_folder_id": file_type_id,
        "clean_project_name": clean_project_name,
        "output_path": f"{project_name}/{branding_name}/{clean_project_name}/{file_type}",
    }


def inspect_output(project_name: str, branding_name: str, file_type: str) -> Dict:
    try:
        dest = resolve_destination(project_name, branding_name, file_type)
        files = list_files_in_folder(dest["svc"], dest["file_type_folder_id"])
        return {
            "ok": True,
            "output_path": dest["output_path"],
            "folder_id": dest["file_type_folder_id"],
            "files": files,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "output_path": f"{project_name}/{branding_name}/.../{file_type}",
            "folder_id": None,
            "files": [],
        }


def upload_bytes(svc, data: bytes, file_name: str, folder_id: str) -> Dict:
    import io

    meta = {"name": file_name, "parents": [folder_id]}

    def create():
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype="application/pdf", resumable=True)
        return svc.files().create(body=meta, media_body=media, fields="id, name, webViewLink, size").execute()

    uploaded = _retry(create, f"upload '{file_name}'")
    return {
        "id": uploaded["id"],
        "name": uploaded["name"],
        "webViewLink": uploaded.get("webViewLink", f"https://drive.google.com/file/d/{uploaded['id']}/view"),
        "size": uploaded.get("size", len(data)),
    }
