"""
PDF Processor - Handles PDF processing and uploading
Reuses existing AI redaction functions from webhook_system
"""

import os
import sys
import time
import tempfile
import gc
from pathlib import Path
from typing import Dict, Callable, Optional, List, Any

sys.path.insert(0, str(Path(__file__).parent))

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


def resolve_upload_destination(
    drive_manager,
    project_name: str,
    branding_name: str,
    file_type: str,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, str]:
    """Resolve Google Drive output folder once for batch uploads."""

    def update_progress(progress: float, message: str):
        if progress_callback:
            progress_callback(progress, message)

    update_progress(0.75, "🔍 Finding upload destination in Google Drive...")

    output_project_id = drive_manager.find_or_create_folder(
        drive_manager.output_folder_id,
        project_name,
    )
    update_progress(0.8, f"✅ Found project: {project_name}")

    output_branding_id = drive_manager.find_or_create_folder(
        output_project_id,
        branding_name,
    )
    update_progress(0.85, f"✅ Found branding: {branding_name}")

    update_progress(0.87, "🔍 Looking for existing project name folder...")

    results = drive_manager.drive_service.files().list(
        q=f"'{output_branding_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
        fields="files(id, name)",
        pageSize=10,
    ).execute()

    project_name_folder = None
    clean_project_name = None
    for folder in results.get("files", []):
        if folder["name"] not in ["Raw", "AI data"]:
            project_name_folder = folder
            clean_project_name = folder["name"]
            break

    if not project_name_folder:
        raise Exception(
            f"Could not find project name folder under {branding_name}. "
            "Expected a folder other than 'Raw' or 'AI data'"
        )

    update_progress(0.9, f"✅ Found existing project folder: {clean_project_name}")

    file_type_id = drive_manager.find_or_create_folder(
        project_name_folder["id"],
        file_type,
    )
    update_progress(0.92, f"✅ Found/created file type: {file_type}")

    return {
        "file_type_folder_id": file_type_id,
        "clean_project_name": clean_project_name,
        "output_path_prefix": (
            f"{project_name}/{branding_name}/{clean_project_name}/{file_type}"
        ),
    }


def process_and_upload_pdf(
    uploaded_file,
    project_name: str,
    project_id: str,
    branding_name: str,
    branding_id: str,
    file_type: str,
    drive_manager,
    include_footer: bool = True,
    progress_callback: Optional[Callable] = None,
    destination: Optional[Dict[str, str]] = None,
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
            
            # Reset file pointer to beginning (important for multiple uploads in same session)
            uploaded_file.seek(0)
            
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            update_progress(0.2, "📥 File saved to temporary location")
            
            # Process PDF
            output_path = os.path.join(temp_dir, f"processed_{uploaded_file.name}")
            
            footer_msg = "watermark + footer" if include_footer else "watermark only"
            update_progress(0.3, f"🎨 Processing PDF ({footer_msg})...")
            
            try:
                # Use existing AI processing
                process_pdf_enhanced(
                    input_path=input_path,
                    output_path=output_path,
                    project_folder_path=f"{branding_name}/{project_name}",
                    include_footer=include_footer
                )
                update_progress(0.6, "✅ PDF processing complete")
            except Exception as process_error:
                raise Exception(f"PDF processing failed: {process_error}")
            
            # Verify processed file exists
            if not os.path.exists(output_path):
                raise Exception("Processed file was not created")
            
            file_size = os.path.getsize(output_path)
            update_progress(0.7, f"📊 Processed file size: {file_size / 1024 / 1024:.2f} MB")

            if destination is None:
                destination = resolve_upload_destination(
                    drive_manager,
                    project_name,
                    branding_name,
                    file_type,
                    progress_callback=progress_callback,
                )

            file_type_id = destination["file_type_folder_id"]
            clean_project_name = destination["clean_project_name"]

            # Upload processed file
            update_progress(0.95, "📤 Uploading processed file to Google Drive...")

            uploaded_file_info = drive_manager.upload_file(
                output_path,
                uploaded_file.name,
                file_type_id,
            )
            
            processing_time = time.time() - start_time
            
            update_progress(1.0, "✅ Upload complete!")
            
            # Build result
            result = {
                'success': True,
                'output_path': f"{destination['output_path_prefix']}/{uploaded_file.name}",
                'file_id': uploaded_file_info['id'],
                'file_name': uploaded_file_info['name'],
                'drive_url': uploaded_file_info.get('webViewLink', 'N/A'),
                'processing_time': processing_time,
                'file_size': file_size
            }
            
            # Cleanup for next upload in same session
            gc.collect()
            
            return result
    
    except Exception as e:
        # Return error result
        processing_time = time.time() - start_time
        # Cleanup even on error
        gc.collect()
        raise Exception(f"Processing failed after {processing_time:.1f}s: {str(e)}")


def process_and_upload_pdfs(
    uploaded_files: List[Any],
    project_name: str,
    project_id: str,
    branding_name: str,
    branding_id: str,
    file_type: str,
    drive_manager,
    include_footer: bool = True,
    progress_callback: Optional[Callable] = None,
) -> List[Dict]:
    """Process and upload multiple PDFs, reusing one Drive destination lookup."""
    if not uploaded_files:
        return []

    total_files = len(uploaded_files)
    results: List[Dict] = []
    errors: List[Dict] = []

    destination = resolve_upload_destination(
        drive_manager,
        project_name,
        branding_name,
        file_type,
    )

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        file_label = f"[{index}/{total_files}] {uploaded_file.name}"

        def file_progress(progress: float, message: str, idx=index, label=file_label):
            if progress_callback:
                overall = ((idx - 1) + progress) / total_files
                progress_callback(overall, f"{label}: {message}")

        try:
            result = process_and_upload_pdf(
                uploaded_file=uploaded_file,
                project_name=project_name,
                project_id=project_id,
                branding_name=branding_name,
                branding_id=branding_id,
                file_type=file_type,
                drive_manager=drive_manager,
                include_footer=include_footer,
                progress_callback=file_progress,
                destination=destination,
            )
            results.append(result)
        except Exception as error:
            errors.append({"file_name": uploaded_file.name, "error": str(error)})

    if errors and not results:
        failed_names = ", ".join(item["file_name"] for item in errors)
        raise Exception(f"All uploads failed: {failed_names}")

    if errors:
        results.append({"partial_errors": errors})

    return results

