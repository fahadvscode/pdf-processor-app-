"""
PDF Processor - Web Interface
Simple, reliable PDF processing with Google Drive integration
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent / "webhook_system"))

from google_drive_helper import DriveManager
from pdf_processor import process_and_upload_pdf

# Page config
st.set_page_config(
    page_title="PDF Processor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PASSWORD PROTECTION
# ============================================================
def check_password():
    """Returns `True` if the user has entered the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "Wintertime2021!":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run, show password input
    if "password_correct" not in st.session_state:
        st.markdown("<h1 style='text-align: center;'>🔒 PDF Processor</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Please enter the password to access this application.</p>", unsafe_allow_html=True)
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password",
            label_visibility="collapsed"
        )
        return False
    # Password not correct, show input + error
    elif not st.session_state["password_correct"]:
        st.markdown("<h1 style='text-align: center;'>🔒 PDF Processor</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Please enter the password to access this application.</p>", unsafe_allow_html=True)
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password",
            label_visibility="collapsed"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

# Check password before showing the app
if not check_password():
    st.stop()  # Do not continue if check_password is not True

# ============================================================
# MAIN APP (Only shown after password is correct)
# ============================================================

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'drive_manager' not in st.session_state:
    st.session_state.drive_manager = None
if 'projects' not in st.session_state:
    st.session_state.projects = None

def initialize_drive():
    """Initialize Google Drive connection"""
    try:
        with st.spinner("🔌 Connecting to Google Drive..."):
            st.session_state.drive_manager = DriveManager()
        
        # Load all projects with progress indicator
        progress_placeholder = st.empty()
        progress_placeholder.info("📂 Loading all projects (this may take a few seconds for 3000+ projects)...")
        
        st.session_state.projects = st.session_state.drive_manager.get_all_projects()
        
        progress_placeholder.success(f"✅ Loaded {len(st.session_state.projects)} projects!")
        
        return True
    except Exception as e:
        st.error(f"❌ Failed to connect to Google Drive: {e}")
        return False

# Main UI
st.markdown('<div class="main-header">🎨 PDF Processor</div>', unsafe_allow_html=True)
st.markdown("**Simple, reliable PDF processing with instant feedback**")

# Sidebar - Connection status
with st.sidebar:
    st.header("🔌 Connection")
    
    if st.session_state.drive_manager is None:
        if st.button("Connect to Google Drive", type="primary"):
            initialize_drive()
    else:
        st.success("✅ Connected to Google Drive")
        if st.button("Refresh Projects"):
            with st.spinner("📂 Refreshing projects list (loading all 3000+ projects)..."):
                st.session_state.projects = st.session_state.drive_manager.get_all_projects()
            st.success(f"✅ Refreshed! Now showing {len(st.session_state.projects)} projects")
            st.rerun()
    
    st.divider()
    
    st.header("📊 Stats")
    if st.session_state.projects:
        st.metric("Projects Available", len(st.session_state.projects))
    
    st.divider()
    
    st.header("ℹ️ About")
    st.markdown("""
    **How it works:**
    1. Select project & branding
    2. Choose file type
    3. Upload PDF
    4. Watch it process
    5. Done! ✨
    
    **No waiting, no webhooks, no complexity.**
    """)

# Main content
if st.session_state.drive_manager is None:
    st.markdown('<div class="info-box">👈 Click "Connect to Google Drive" in the sidebar to get started</div>', unsafe_allow_html=True)
    
    # Show features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🚀 Instant")
        st.markdown("Upload → Process → Done in seconds")
    with col2:
        st.markdown("### 👀 Visible")
        st.markdown("See exactly what's happening in real-time")
    with col3:
        st.markdown("### 🛡️ Reliable")
        st.markdown("No crashes, no timing issues, no complexity")

else:
    # Step 1: Select Project
    st.header("1️⃣ Select Project")
    
    # Search box for project
    search_query = st.text_input("🔍 Search project by name or ID", placeholder="Type to search...")
    
    # Filter projects based on search
    filtered_projects = st.session_state.projects
    if search_query:
        filtered_projects = [p for p in st.session_state.projects 
                           if search_query.lower() in p['name'].lower()]
    
    if not filtered_projects:
        st.warning("No projects found matching your search")
    else:
        # Show number of results
        if search_query:
            st.info(f"Found {len(filtered_projects)} project(s)")
        
        # Check for duplicate names and show IDs if needed
        name_counts = {}
        for p in filtered_projects:
            name = p['name']
            name_counts[name] = name_counts.get(name, 0) + 1
        
        has_duplicates = any(count > 1 for count in name_counts.values())
        
        # Create display options (show ID if duplicates exist)
        if has_duplicates:
            project_options = [f"{p['name']} (ID: {p['id'][:8]}...)" for p in filtered_projects]
        else:
            project_options = [p['name'] for p in filtered_projects]
        
        # Project selector
        selected_option = st.selectbox(
            "Select project",
            options=project_options,
            label_visibility="collapsed"
        )
        
        # Get selected project details
        # Extract the index from the selected option
        selected_index = project_options.index(selected_option)
        selected_project = filtered_projects[selected_index]
        selected_project_name = selected_project['name']
        
        # Show warning if there are duplicates
        if has_duplicates and name_counts.get(selected_project_name, 0) > 1:
            st.warning(f"⚠️ Multiple projects with name '{selected_project_name}' found. Selected: {selected_project['id']}")
        
        st.divider()
        
        # Step 2: Select Branding
        st.header("2️⃣ Select Branding")
        
        brandings = st.session_state.drive_manager.get_brandings(selected_project['id'])
        branding_names = [b['name'] for b in brandings]
        
        selected_branding_name = st.selectbox(
            "Select branding",
            options=branding_names,
            label_visibility="collapsed"
        )
        
        selected_branding = next(b for b in brandings if b['name'] == selected_branding_name)
        
        st.divider()
        
        # Step 3: Select File Type
        st.header("3️⃣ Select File Type")
        
        # Get actual file types from Raw folder in Drive
        with st.spinner("📂 Loading file types from Drive..."):
            file_types = st.session_state.drive_manager.get_file_types_from_raw_folder(selected_branding['id'])
        
        if not file_types:
            st.warning("⚠️ No file type folders found in Raw folder. Please ensure the Raw folder exists with subfolders.")
            st.info("💡 Tip: The Raw folder should contain subfolders like 'Floor Plans', 'Renderings', etc.")
            selected_type = None
        else:
            selected_type = st.selectbox(
                "Select file type",
                options=file_types,
                label_visibility="collapsed"
            )
            st.info(f"✅ Found {len(file_types)} file type(s) in Raw folder")
        
        st.divider()
        
        # Step 4: Upload File
        st.header("4️⃣ Upload PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Show file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.success(f"✅ File selected: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
            
            # Check if file type is selected
            if not selected_type:
                st.error("❌ Please select a file type first (Step 3)")
                st.stop()
            
            # Process button
            if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                try:
                    # Processing
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Download to temp
                    status_text.text("📥 Preparing file...")
                    progress_bar.progress(0.1)
                    
                    # Step 2: Process PDF
                    status_text.text("🎨 Processing PDF (removing text, adding watermark)...")
                    progress_bar.progress(0.3)
                    
                    result = process_and_upload_pdf(
                        uploaded_file=uploaded_file,
                        project_name=selected_project['name'],
                        project_id=selected_project['id'],
                        branding_name=selected_branding_name,
                        branding_id=selected_branding['id'],
                        file_type=selected_type,
                        drive_manager=st.session_state.drive_manager,
                        progress_callback=lambda p, msg: (
                            progress_bar.progress(0.3 + (p * 0.6)),
                            status_text.text(msg)
                        )
                    )
                    
                    # Complete
                    progress_bar.progress(1.0)
                    status_text.empty()
                    
                    # Success message
                    drive_link = result.get('drive_url', 'N/A')
                    file_size_mb = result.get('file_size', 0) / (1024 * 1024)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Success!</h3>
                        <p><strong>File processed and uploaded successfully!</strong></p>
                        <p>📁 Location: {result['output_path']}</p>
                        <p>📄 File: {uploaded_file.name} ({file_size_mb:.2f} MB)</p>
                        <p>⏱️ Processing time: {result['processing_time']:.1f}s</p>
                        <p>🔗 <a href="{drive_link}" target="_blank">👉 Click here to view file in Google Drive</a></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Also show as a button for better visibility
                    if drive_link != 'N/A':
                        st.link_button("🔗 Open File in Google Drive", drive_link)
                    
                    st.balloons()
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-box">
                        <h3>❌ Error</h3>
                        <p>{str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show error details in expander
                    with st.expander("🔍 Error Details"):
                        st.code(str(e))

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    PDF Processor v2.0 - Web Interface | No webhooks, no complexity, just works ✨
</div>
""", unsafe_allow_html=True)

