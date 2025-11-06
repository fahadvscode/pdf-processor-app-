# PDF Processor - Web Interface 🎨

Simple, reliable PDF processing with a beautiful web interface. No webhooks, no complexity - just upload and process.

## Why This Instead of Webhooks?

| Webhook System ❌ | Web Interface ✅ |
|-------------------|------------------|
| Complex setup | Run 1 command |
| Needs 24/7 server | Run only when needed |
| Crashes & timing issues | Stable & reliable |
| Can't see what's happening | Real-time progress |
| Hard to debug | Errors shown immediately |
| Must configure ngrok, tokens, etc. | Just click and upload |

## Features

✅ **Search Projects** - Autocomplete search with 3000+ projects  
✅ **Visual Interface** - See exactly what's happening  
✅ **Real-Time Progress** - Watch your file process  
✅ **Instant Feedback** - Success/errors shown immediately  
✅ **No Setup** - Reuses existing Google Drive auth  
✅ **Drag & Drop** - Easy file upload  
✅ **Reliable** - No crashes, no webhook issues  

## Quick Start

### 1. Install Dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### 3. Use the App

1. **Connect to Google Drive** (uses existing credentials from webhook_system)
2. **Search for your project** - Type to filter from 3000+ projects
3. **Select branding** - Choose "Precon Factory", "Fahad Javed", etc.
4. **Choose file type** - Floor Plans, Renderings, etc.
5. **Upload PDF** - Drag & drop or click to upload
6. **Click "Process PDF"** - Watch it process in real-time
7. **Done!** - File automatically uploaded to Google Drive

## How It Works

```
Upload File
    ↓
Save to temp folder
    ↓
Process PDF (AI redaction, watermark)
    ↓
Find output location in Drive
    ↓
Upload processed file
    ↓
Show success message + Drive link
```

## Folder Structure

The app maintains the same folder structure:

**Input:** `ProjectName/BrandingName/Raw/FileType/`  
**Output:** `ProjectName/BrandingName/AI data/FileType/`

## Benefits Over Webhook System

1. **No Server Required** - Run on your laptop when you need it
2. **No Webhook Complexity** - No ngrok, no tokens, no timing issues
3. **Visual Feedback** - See progress bar, status messages
4. **Immediate Error Handling** - If something fails, you see it right away
5. **Easy to Debug** - Errors shown in UI with details
6. **No 24/7 Monitoring** - Just run when you need to process files
7. **Simpler Maintenance** - No crashes, no restarts, no SSH

## Switching Between Systems

### Use Web Interface (This):
- **When:** You're actively uploading files
- **How:** Run `streamlit run app.py`
- **Benefit:** Full control, visual feedback

### Use Webhook System:
- **When:** You want automatic 24/7 processing
- **How:** Run the monitor on DigitalOcean
- **Benefit:** Automatic, hands-off

Both systems work independently. Keep both!

## Troubleshooting

### "Failed to connect to Google Drive"
- Make sure `credentials.json` exists in `webhook_system/` folder
- Run the app once to authenticate (it will open a browser)

### "No module named 'ai_enhanced_redactor'"
- The app imports from `../webhook_system/`
- Make sure the webhook_system folder exists

### "Port already in use"
- Streamlit is probably already running
- Close other Streamlit apps or use: `streamlit run app.py --server.port 8502`

## Tech Stack

- **Streamlit** - Web interface (no HTML/CSS/JS needed!)
- **Google Drive API** - File operations
- **PyMuPDF** - PDF processing
- **Existing AI Redactor** - Reuses your webhook system's processing

## File Structure

```
streamlit_app/
├── app.py                    # Main Streamlit app
├── google_drive_helper.py    # Google Drive operations
├── pdf_processor.py          # PDF processing & upload
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Next Steps

Want to deploy this to the cloud? You can:

1. **Streamlit Cloud** (Free) - Deploy with one click, share URL with anyone
2. **Heroku** - Deploy as web app
3. **Keep it Local** - Run on your laptop when needed

For now, running locally is simplest and works great!

---

**Questions?** Just ask! This is much simpler than the webhook system and actually works reliably.

