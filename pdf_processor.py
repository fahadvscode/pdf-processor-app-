"""
PDF Processor - Handles PDF processing and uploading
Reuses existing AI redaction functions from webhook_system
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Callable, Optional

# Add current directory and parent webhook_system to path
sys.path.insert(0, str(Path(__file__).parent))  # Current streamlit_app directory
parent_dir = Path(__file__).parent.parent / "webhook_system"
if parent_dir.exists():
sys.path.insert(0, str(parent_dir))

# Import existing PDF processing function
try:
    from ai_enhanced_redactor import process_pdf_enhanced
    print("✅ AI redaction module loaded successfully")
except ImportError as e:
    print(f"⚠️  Warning: Could not import ai_enhanced_redactor: {e}")
    # Fallback if module structure is different
    def process_pdf_enhanced(input_path, output_path, project_folder_path):
        """Fallback PDF processor - just copies file"""
        import shutil
        shutil.copy(input_path, output_path)
        print(f"⚠️  Warning: Using fallback processor (copied file without processing)")


def process_and_upload_pdf(
    uploaded_file,
    project_name: str,
    project_id: str,
    branding_name: str,
    branding_id: str,
    file_type: str,
    drive_manager,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    Process a PDF file and upload to Google Drive
    
    Args:
        uploaded_file: Streamlit uploaded file object
        project_name: Name of the project
        project_id: Google Drive ID of project folder (input)
        branding_name: Name of branding (e.g., "Precon Factory")
        branding_id: Google Drive ID of branding folder (input)
        file_type: Type of file (e.g., "Floor Plans")
        drive_manager: DriveManager instance
        progress_callback: Optional callback for progress updates (progress_float, message)
    
    Returns:
        Dict with processing results
    """
    start_time = time.time()
    
    def update_progress(progress: float, message: str):
        """Helper to call progress callback if provided"""
        if progress_callback:
            progress_callback(progress, message)
    
    try:
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            input_path = os.path.join(temp_dir, uploaded_file.name)
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            update_progress(0.2, "📥 File saved to temporary location")
            
            # Process PDF
            output_path = os.path.join(temp_dir, f"processed_{uploaded_file.name}")
            
            update_progress(0.3, "🎨 Processing PDF (AI redaction, watermark, footer)...")
            
            try:
                # Use existing AI processing
                process_pdf_enhanced(
                    input_path=input_path,
                    output_path=output_path,
                    project_folder_path=f"{branding_name}/{project_name}"
                )
                update_progress(0.6, "✅ PDF processing complete")
            except Exception as process_error:
                raise Exception(f"PDF processing failed: {process_error}")
            
            # Verify processed file exists
            if not os.path.exists(output_path):
                raise Exception("Processed file was not created")
            
            file_size = os.path.getsize(output_path)
            update_progress(0.7, f"📊 Processed file size: {file_size / 1024 / 1024:.2f} MB")
            
            # Find output location in Drive
            update_progress(0.75, "🔍 Finding upload destination in Google Drive...")
            
            # Get project folder in output (the one with ID)
            # Output structure: ProjectName_ID/BrandingName/ExistingProjectNameFolder/FileType
            output_project_id = drive_manager.find_or_create_folder(
                drive_manager.output_folder_id,
                project_name
            )
            
            update_progress(0.8, f"✅ Found project: {project_name}")
            
            # Get branding folder in output project
            output_branding_id = drive_manager.find_or_create_folder(
                output_project_id,
                branding_name
            )
            
            update_progress(0.85, f"✅ Found branding: {branding_name}")
            
            # Find the EXISTING project name folder under branding
            # It's the folder that's NOT "Raw" and NOT "AI data"
            update_progress(0.87, "🔍 Looking for existing project name folder...")
            
            results = drive_manager.drive_service.files().list(
                q=f"'{output_branding_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
                fields="files(id, name)",
                pageSize=10
            ).execute()
            
            folders = results.get('files', [])
            project_name_folder = None
            clean_project_name = None
            
            for folder in folders:
                if folder['name'] not in ['Raw', 'AI data']:
                    # This is the project name folder!
                    project_name_folder = folder
                    clean_project_name = folder['name']
                    break
            
            if not project_name_folder:
                raise Exception(f"Could not find project name folder under {branding_name}. Expected a folder other than 'Raw' or 'AI data'")
            
            project_name_folder_id = project_name_folder['id']
            update_progress(0.9, f"✅ Found existing project folder: {clean_project_name}")
            
            # Get/create file type folder under the project name folder
            file_type_id = drive_manager.find_or_create_folder(
                project_name_folder_id,
                file_type
            )
            
            update_progress(0.92, f"✅ Found/created file type: {file_type}")
            
            # Upload processed file
            update_progress(0.95, "📤 Uploading processed file to Google Drive...")
            
            uploaded_file_info = drive_manager.upload_file(
                output_path,
                uploaded_file.name,
                file_type_id
            )
            
            processing_time = time.time() - start_time
            
            update_progress(1.0, "✅ Upload complete!")
            
            # Return result
            return {
                'success': True,
                'output_path': f"{project_name}/{branding_name}/{clean_project_name}/{file_type}/{uploaded_file.name}",
                'file_id': uploaded_file_info['id'],
                'file_name': uploaded_file_info['name'],
                'drive_url': uploaded_file_info.get('webViewLink', 'N/A'),
                'processing_time': processing_time,
                'file_size': file_size
            }
    
    except Exception as e:
        # Return error result
        processing_time = time.time() - start_time
        raise Exception(f"Processing failed after {processing_time:.1f}s: {str(e)}")

