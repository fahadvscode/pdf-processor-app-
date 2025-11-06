"""
Helper script to extract Google Drive token for Streamlit Cloud deployment
Run this locally to get the token JSON that you'll paste into Streamlit Cloud secrets
"""

import pickle
import json
from pathlib import Path

def extract_token():
    """Extract token from pickle file and format for Streamlit secrets"""
    
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent / "webhook_system" / "token.pickle",
        Path(__file__).parent / "token.pickle",
        Path("../webhook_system/token.pickle"),
        Path("token.pickle"),
    ]
    
    token_path = None
    for path in possible_paths:
        if path.exists():
            token_path = path
            break
    
    if not token_path:
        print("❌ Error: token.pickle not found!")
        print("\nSearched locations:")
        for path in possible_paths:
            print(f"  - {path.absolute()}")
        print("\n💡 Run the Streamlit app locally first to generate the token.")
        return
    
    print(f"✅ Found token at: {token_path.absolute()}\n")
    
    # Load the token
    with open(token_path, 'rb') as f:
        creds = pickle.load(f)
    
    # Convert to JSON format
    token_dict = {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes
    }
    
    # Pretty print JSON
    token_json = json.dumps(token_dict, indent=2)
    
    print("=" * 60)
    print("📋 COPY THIS TO STREAMLIT CLOUD SECRETS:")
    print("=" * 60)
    print("\n[token]")
    print('token = """')
    print(token_json)
    print('"""')
    print("\n")
    print("=" * 60)
    print("📌 INSTRUCTIONS:")
    print("=" * 60)
    print("1. Go to your Streamlit Cloud app dashboard")
    print("2. Click '⚙️ Settings' → 'Secrets'")
    print("3. Paste the above content (including [token] section)")
    print("4. Add the folder IDs:")
    print("\nDRIVE_INPUT_FOLDER_ID = \"1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88\"")
    print("DRIVE_OUTPUT_FOLDER_ID = \"1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88\"")
    print("\n5. Click 'Save'")
    print("6. Your app will restart and connect to Google Drive!")
    print("=" * 60)
    
    # Also save to a file for easy copying
    output_file = Path(__file__).parent / "streamlit_secrets.txt"
    with open(output_file, 'w') as f:
        f.write("[token]\n")
        f.write('token = """\n')
        f.write(token_json)
        f.write('\n"""\n\n')
        f.write("DRIVE_INPUT_FOLDER_ID = \"1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88\"\n")
        f.write("DRIVE_OUTPUT_FOLDER_ID = \"1oRwJHz2eeyDTm0hn6k6XlcII8o67BY88\"\n")
    
    print(f"\n✅ Also saved to: {output_file}")
    print("   You can copy from this file!\n")

if __name__ == "__main__":
    extract_token()

