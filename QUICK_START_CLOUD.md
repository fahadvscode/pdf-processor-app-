# 🚀 Quick Start: Deploy to Streamlit Cloud

## 3 Simple Steps

### Step 1: Get Your Token 🔑

Run this on your Mac (where you have the working app):

```bash
cd "/Users/fahadjaved/Documents/pdf cleanup all files backup/number 3/pdf cleanup 2/pdf cleanup/streamlit_app"
python extract_token_for_cloud.py
```

This will create `streamlit_secrets.txt` with your Google Drive credentials.

---

### Step 2: Push to GitHub 📤

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Add PDF Processor Streamlit App"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

---

### Step 3: Deploy to Streamlit Cloud ☁️

1. **Go to:** https://share.streamlit.io/

2. **Sign in** with GitHub

3. **Click "New app"**

4. **Fill in:**
   - Repository: `YOUR_USERNAME/YOUR_REPO`
   - Branch: `main`
   - Main file: `streamlit_app/app.py`
   - App URL: Choose your subdomain (e.g., `pdf-processor`)

5. **Click "Deploy"**

6. **Add Secrets:**
   - In dashboard, click **⚙️ Settings → Secrets**
   - Open `streamlit_secrets.txt` and copy ALL content
   - Paste into Streamlit secrets box
   - Click **Save**

7. **Done!** 🎉

Your app will be live at:
```
https://your-app-name.streamlit.app
```

---

## 🌍 Access From Anywhere

- ✅ Use from any country
- ✅ Any device (laptop, phone, tablet)
- ✅ Share link with team
- ✅ HTTPS secure
- ✅ No VPN or firewall issues

---

## 🆘 Troubleshooting

**"No token found"**
- Run `extract_token_for_cloud.py` first
- Make sure you've run the app locally at least once

**"Module not found"**
- Check `requirements.txt` is in repo
- Streamlit Cloud will auto-install

**"Auth failed"**
- Copy secrets EXACTLY from `streamlit_secrets.txt`
- Make sure no extra spaces

---

## 📞 Need Help?

See full guide: `DEPLOYMENT_GUIDE.md`


