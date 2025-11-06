"""
Google Drive Helper - Simplified interface for Streamlit app
Reuses existing Google Drive authentication and functions
"""

import os
import sys
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Try to import streamlit for cloud deployment
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Add parent directory to access webhook_system modules
parent_dir = Path(__file__).parent.parent / "webhook_system"
sys.path.insert(0, str(parent_dir))

SCOPES = ['https://www.googleapis.com/auth/drive']

# Configuration (same as webhook system)
# Try to get from Streamlit secrets first, then environment, then default
def get_folder_id(key, default):
    """Get folder ID from Streamlit secrets, env var, or default"""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    return os.getenv(key, default)

DRIVE_INPUT_FOLDER_ID = get_folder_id('DRIVE_INPUT_FOLDER_ID', '1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88')
DRIVE_OUTPUT_FOLDER_ID = get_folder_id('DRIVE_OUTPUT_FOLDER_ID', '1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88')  # Same as input - we route to AI data subfolder

class DriveManager:
    """Manages Google Drive operations for the Streamlit app"""
    
    def __init__(self):
        """Initialize Drive service"""
        self.creds = self._get_credentials()
        self.drive_service = build('drive', 'v3', credentials=self.creds)
        self.input_folder_id = DRIVE_INPUT_FOLDER_ID
        self.output_folder_id = DRIVE_OUTPUT_FOLDER_ID
    
    def _get_credentials(self):
        """Get Google Drive credentials (supports both local and Streamlit Cloud)"""
        creds = None
        
        # Detect if running on Streamlit Cloud
        import os
        is_streamlit_cloud = (
            os.getenv('STREAMLIT_SHARING_MODE') or 
            os.getenv('STREAMLIT_SERVER_PORT') or
            not Path(__file__).parent.parent.joinpath("webhook_system").exists()
        )
        
        # Check if running on Streamlit Cloud with secrets
        if HAS_STREAMLIT and hasattr(st, 'secrets'):
            try:
                # Check if we have a pre-stored token in secrets (Streamlit Cloud)
                if 'token' in st.secrets:
                    token_data = st.secrets['token']
                    
                    # Debug: Show what we got
                    debug_info = {
                        'type': str(type(token_data)),
                        'is_dict': isinstance(token_data, dict),
                        'is_str': isinstance(token_data, str),
                    }
                    if isinstance(token_data, dict):
                        debug_info['keys'] = list(token_data.keys())
                        if 'token' in token_data:
                            debug_info['token_type'] = str(type(token_data['token']))
                            debug_info['token_preview'] = str(token_data['token'])[:100]
                    
                    # Parse the token based on format
                    token_info = None
                    
                    if isinstance(token_data, dict):
                        if 'token' in token_data:
                            # Format: [token] token = "{...json...}"
                            token_str = token_data['token']
                            if isinstance(token_str, str):
                                # Clean and parse JSON string
                                token_str = token_str.strip()
                                if token_str.startswith('{') and token_str.endswith('}'):
                                    try:
                                        token_info = json.loads(token_str)
                                    except json.JSONDecodeError as e:
                                        raise Exception(f"Failed to parse token JSON: {e}\nDebug: {debug_info}")
                                else:
                                    raise Exception(f"Token string doesn't look like JSON: {token_str[:100]}...\nDebug: {debug_info}")
                            else:
                                # token_data itself is the dict
                                token_info = token_data
                        else:
                            # Token data is already the dict we need (flat structure)
                            token_info = token_data
                    elif isinstance(token_data, str):
                        # Token data is a JSON string
                        try:
                            token_info = json.loads(token_data)
                        except json.JSONDecodeError as e:
                            raise Exception(f"Failed to parse token JSON string: {e}\nDebug: {debug_info}")
                    
                    if not token_info:
                        raise Exception(f"Could not parse token from secrets. Debug info: {debug_info}")
                    
                    # Verify required fields
                    required_fields = ['token', 'refresh_token', 'token_uri', 'client_id', 'client_secret']
                    missing_fields = [f for f in required_fields if f not in token_info]
                    if missing_fields:
                        raise Exception(
                            f"Token missing required fields: {missing_fields}\n"
                            f"Token keys found: {list(token_info.keys())}\n"
                            f"Token preview: {str(token_info)[:200]}..."
                        )
                    
                    creds = Credentials.from_authorized_user_info(token_info, SCOPES)
                    
                    # Refresh if expired (note: can't refresh if refresh_token is null)
                    if creds and creds.expired and creds.refresh_token:
                        try:
                            creds.refresh(Request())
                        except Exception as refresh_error:
                            # If refresh fails, return the creds anyway and hope they work
                            print(f"Token refresh failed: {refresh_error}")
                    
                    return creds
                else:
                    if is_streamlit_cloud:
                        raise Exception(
                            "Token not found in Streamlit secrets. "
                            "Please add your Google Drive credentials to Streamlit Cloud secrets."
                        )
                    
            except Exception as e:
                # If on Streamlit Cloud, don't fall back to local files
                if is_streamlit_cloud:
                    raise Exception(
                        f"Failed to load credentials from Streamlit secrets: {e}\n"
                        f"Secrets keys available: {list(st.secrets.keys()) if hasattr(st, 'secrets') else 'None'}\n"
                        "Please check your secrets configuration in Streamlit Cloud."
                    )
                # Fall through to local method only if not on Streamlit Cloud
                print(f"Streamlit secrets failed: {e}. Trying local credentials...")
        
        # Local development: Use token.pickle
        if is_streamlit_cloud:
            raise Exception(
                "Running on Streamlit Cloud but no secrets configured. "
                "Please add your Google Drive credentials to Streamlit Cloud secrets."
            )
        
        token_path = Path(__file__).parent.parent / "webhook_system" / "token.pickle"
        credentials_path = Path(__file__).parent.parent / "webhook_system" / "credentials.json"
        
        # Load existing token
        if token_path.exists():
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # Refresh or get new token
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not credentials_path.exists():
                    raise FileNotFoundError(
                        f"credentials.json not found at {credentials_path}\n"
                        "Please copy it from the webhook_system folder"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save the token
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        return creds
    
    def get_all_projects(self) -> List[Dict]:
        """Get all project folders from input folder (with pagination for 3000+ projects)"""
        try:
            all_projects = []
            page_token = None
            
            while True:
                # Build query
                query_params = {
                    'q': f"'{self.input_folder_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
                    'fields': "nextPageToken, files(id, name)",
                    'pageSize': 1000,  # Maximum page size
                    'orderBy': 'name'
                }
                
                # Add page token if we're continuing pagination
                if page_token:
                    query_params['pageToken'] = page_token
                
                results = self.drive_service.files().list(**query_params).execute()
                
                # Add projects from this page
                projects = results.get('files', [])
                all_projects.extend(projects)
                
                # Check if there are more pages
                page_token = results.get('nextPageToken')
                if not page_token:
                    break  # No more pages
            
            # Deduplicate by project ID (in case of duplicates)
            seen_ids = set()
            unique_projects = []
            for project in all_projects:
                project_id = project.get('id')
                if project_id and project_id not in seen_ids:
                    seen_ids.add(project_id)
                    unique_projects.append(project)
            
            # Sort by name for consistent display
            unique_projects.sort(key=lambda x: x.get('name', '').lower())
            
            return unique_projects
        except Exception as e:
            raise Exception(f"Failed to get projects: {e}")
    
    def get_brandings(self, project_id: str) -> List[Dict]:
        """Get branding folders for a project"""
        try:
            results = self.drive_service.files().list(
                q=f"'{project_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
                fields="files(id, name)",
                pageSize=100
            ).execute()
            
            brandings = results.get('files', [])
            return brandings
        except Exception as e:
            raise Exception(f"Failed to get brandings: {e}")
    
    def get_file_types_from_raw_folder(self, branding_id: str) -> List[str]:
        """Get actual file type subfolders from Raw folder in Drive"""
        try:
            # Find "Raw" folder inside branding
            raw_folder_results = self.drive_service.files().list(
                q=f"name='Raw' and '{branding_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
                fields="files(id, name)",
                pageSize=1
            ).execute()
            
            raw_folders = raw_folder_results.get('files', [])
            if not raw_folders:
                # Raw folder doesn't exist yet, return empty list
                return []
            
            raw_folder_id = raw_folders[0]['id']
            
            # Get all subfolders inside Raw (these are the file types)
            file_types = []
            page_token = None
            
            while True:
                query_params = {
                    'q': f"'{raw_folder_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
                    'fields': "nextPageToken, files(id, name)",
                    'pageSize': 100,
                    'orderBy': 'name'
                }
                
                if page_token:
                    query_params['pageToken'] = page_token
                
                results = self.drive_service.files().list(**query_params).execute()
                
                folders = results.get('files', [])
                file_types.extend([f['name'] for f in folders])
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            # Sort and return
            file_types.sort()
            return file_types
            
        except Exception as e:
            # If there's an error, return empty list (will show fallback)
            return []
    
    def find_or_create_folder(self, parent_id: str, folder_name: str) -> str:
        """Find or create a folder"""
        try:
            # Try to find existing folder
            results = self.drive_service.files().list(
                q=f"name='{folder_name}' and '{parent_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
                fields="files(id)",
                pageSize=1
            ).execute()
            
            files = results.get('files', [])
            if files:
                return files[0]['id']
            
            # Create new folder
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]
            }
            folder = self.drive_service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            return folder['id']
        except Exception as e:
            raise Exception(f"Failed to find/create folder {folder_name}: {e}")
    
    def upload_file(self, local_path: str, file_name: str, folder_id: str) -> Dict:
        """Upload a file to Google Drive and verify it exists"""
        try:
            from googleapiclient.http import MediaFileUpload
            import os
            
            # Verify file exists locally
            if not os.path.exists(local_path):
                raise Exception(f"Local file not found: {local_path}")
            
            file_size = os.path.getsize(local_path)
            
            file_metadata = {
                'name': file_name,
                'parents': [folder_id]
            }
            
            media = MediaFileUpload(
                local_path,
                mimetype='application/pdf',
                resumable=True
            )
            
            # Upload file
            uploaded_file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink, parents, size'
            ).execute()
            
            file_id = uploaded_file.get('id')
            
            # Verify file actually exists in Drive
            try:
                verify_file = self.drive_service.files().get(
                    fileId=file_id,
                    fields='id, name, parents, webViewLink, size'
                ).execute()
                
                # Verify it's in the correct folder
                file_parents = verify_file.get('parents', [])
                if folder_id not in file_parents:
                    raise Exception(f"File uploaded but not in expected folder. Parents: {file_parents}")
                
                return {
                    'id': verify_file['id'],
                    'name': verify_file['name'],
                    'webViewLink': verify_file.get('webViewLink', f'https://drive.google.com/file/d/{file_id}/view'),
                    'parents': file_parents,
                    'size': verify_file.get('size', file_size)
                }
            except Exception as verify_error:
                raise Exception(f"Upload succeeded but verification failed: {verify_error}")
            
        except Exception as e:
            raise Exception(f"Failed to upload file: {e}")
    
    def download_file(self, file_id: str, local_path: str) -> bool:
        """Download a file from Google Drive"""
        try:
            from googleapiclient.http import MediaIoBaseDownload
            
            request = self.drive_service.files().get_media(fileId=file_id)
            
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            
            return True
        except Exception as e:
            raise Exception(f"Failed to download file: {e}")

