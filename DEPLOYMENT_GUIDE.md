# 🚀 Streamlit Cloud Deployment Guide

## Step-by-Step Instructions

### 1️⃣ Prepare Your Repository

**Option A: Create New Repository**
```bash
cd "/Users/fahadjaved/Documents/pdf cleanup all files backup/number 3/pdf cleanup 2/pdf cleanup/streamlit_app"

# Initialize git
git init

# Add files
git add .
git commit -m "Initial commit - PDF Processor Streamlit App"

# Create repo on GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/pdf-processor.git
git branch -M main
git push -u origin main
```

**Option B: Add to Existing Repository**
```bash
# Just push the streamlit_app folder to your existing repo
git add streamlit_app/
git commit -m "Add Streamlit PDF Processor"
git push
```

---

### 2️⃣ Deploy to Streamlit Cloud

1. **Go to:** https://share.streamlit.io/

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Configure deployment:**
   - **Repository:** Select your GitHub repo
   - **Branch:** `main` (or your branch name)
   - **Main file path:** `streamlit_app/app.py` (or just `app.py` if in root)
   - **App URL:** Choose your subdomain (e.g., `pdf-processor`)

5. **Click "Deploy"**

---

### 3️⃣ Add Google Drive Credentials

**IMPORTANT:** You need to add your Google Drive credentials as secrets.

#### Method 1: Using OAuth Credentials (Recommended)

1. In Streamlit Cloud dashboard, click **"⚙️ Settings"**
2. Click **"Secrets"**
3. Copy content from your `credentials.json` file:

```toml
[credentials]
web = """
{
  "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
  "project_id": "your-project-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_secret": "YOUR_CLIENT_SECRET",
  "redirect_uris": ["http://localhost"]
}
"""

# Folder IDs
DRIVE_INPUT_FOLDER_ID = "1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88"
DRIVE_OUTPUT_FOLDER_ID = "1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88"
```

4. Click **"Save"**

#### Method 2: Using Token (If Already Authenticated)

If you have `token.pickle`, you'll need to re-authenticate on Streamlit Cloud or convert it to service account.

---

### 4️⃣ Update Code for Streamlit Cloud

The `google_drive_helper.py` needs a small update to read credentials from Streamlit secrets.

**Add this function to handle Streamlit Cloud secrets:**

```python
def _get_credentials(self):
    """Get Google Drive credentials"""
    creds = None
    
    # Check if running on Streamlit Cloud
    if hasattr(st, 'secrets') and 'credentials' in st.secrets:
        # Use Streamlit secrets
        import json
        credentials_info = json.loads(st.secrets['credentials']['web'])
        # ... handle OAuth flow with secrets
    else:
        # Use local token.pickle (existing code)
        token_path = Path(__file__).parent.parent / "webhook_system" / "token.pickle"
        # ... existing local code
```

---

### 5️⃣ Access Your App

Once deployed, your app will be available at:

```
https://your-app-name.streamlit.app
```

Example:
```
https://pdf-processor.streamlit.app
```

✅ **Accessible worldwide**
✅ **HTTPS secure**
✅ **No installation needed**
✅ **Share link with anyone**

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: ai_enhanced_redactor"

The app imports from `webhook_system` folder. Make sure:
- Include `webhook_system` folder in your repo
- Or adjust imports to include the redactor in streamlit_app

### "Google Auth Failed"

- Double-check secrets are correctly formatted
- No extra spaces or quotes
- Valid JSON format

### "Port Already in Use"

Not an issue on Streamlit Cloud - only for local development.

### App Crashes on Startup

- Check logs in Streamlit Cloud dashboard
- Verify all dependencies in requirements.txt
- Check secrets are configured

---

## 📦 Required Files Structure

```
your-repo/
├── streamlit_app/
│   ├── .streamlit/
│   │   └── config.toml          ← Theme & settings
│   ├── app.py                   ← Main app
│   ├── google_drive_helper.py   ← Drive operations
│   ├── pdf_processor.py         ← PDF processing
│   └── requirements.txt         ← Dependencies
│
└── webhook_system/              ← Include if using imports
    └── ai_enhanced_redactor.py
```

---

## 🎯 Quick Checklist

- [ ] Code pushed to GitHub
- [ ] Streamlit Cloud account created
- [ ] App deployed from GitHub
- [ ] Google credentials added to secrets
- [ ] Folder IDs configured
- [ ] App loads without errors
- [ ] Can connect to Google Drive
- [ ] Can process and upload PDFs

---

## 🌐 Sharing Your App

Once deployed, share the URL with:
- ✅ Your team
- ✅ Clients
- ✅ Anyone with the link

**Optional:** You can password-protect it in Streamlit Cloud settings.

---

## 💡 Next Steps After Deployment

1. **Test thoroughly** - Upload a few PDFs
2. **Monitor logs** - Check for any errors
3. **Add custom domain** (optional) - Use your own domain like pdf.fahadsold.com
4. **Set up notifications** - Get alerts if app goes down

---

**Need help?** Check Streamlit Cloud docs: https://docs.streamlit.io/streamlit-community-cloud


