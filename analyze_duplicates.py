"""
Analyze duplicate project folders in Google Drive
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to access modules
parent_dir = Path(__file__).parent.parent / "webhook_system"
sys.path.insert(0, str(parent_dir))

from google_drive_helper import DriveManager

def analyze_duplicates():
    """Analyze and report duplicate project folders"""
    print("🔍 Analyzing project folders for duplicates...")
    print("=" * 60)
    
    try:
        # Connect to Drive
        print("📡 Connecting to Google Drive...")
        drive_manager = DriveManager()
        
        # Get all projects
        print("📂 Loading all projects...")
        projects = drive_manager.get_all_projects()
        
        print(f"✅ Found {len(projects)} total projects\n")
        
        # Group by name
        name_groups = defaultdict(list)
        for project in projects:
            name = project['name']
            name_groups[name].append(project)
        
        # Find duplicates
        duplicates = {name: projs for name, projs in name_groups.items() if len(projs) > 1}
        
        # Report results
        print("=" * 60)
        print("📊 DUPLICATE ANALYSIS RESULTS")
        print("=" * 60)
        
        if not duplicates:
            print("✅ No duplicate project names found!")
            print(f"   All {len(projects)} projects have unique names.")
        else:
            print(f"⚠️  Found {len(duplicates)} project name(s) with duplicates:\n")
            
            total_duplicates = 0
            for name, projs in sorted(duplicates.items()):
                count = len(projs)
                total_duplicates += (count - 1)  # Count extra copies
                
                print(f"📁 '{name}'")
                print(f"   Appears {count} times:")
                for i, proj in enumerate(projs, 1):
                    print(f"      {i}. ID: {proj['id']}")
                print()
            
            print("=" * 60)
            print(f"📈 SUMMARY:")
            print(f"   Total projects: {len(projects)}")
            print(f"   Unique project names: {len(name_groups)}")
            print(f"   Duplicate names: {len(duplicates)}")
            print(f"   Extra folders (duplicates): {total_duplicates}")
            print(f"   Projects with unique names: {len(projects) - total_duplicates - len(duplicates)}")
            print("=" * 60)
        
        # Show top 10 most common names (if any duplicates)
        if duplicates:
            print("\n🔝 TOP 10 MOST DUPLICATED PROJECT NAMES:")
            sorted_dups = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
            for i, (name, projs) in enumerate(sorted_dups[:10], 1):
                print(f"   {i}. '{name}' - {len(projs)} copies")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_duplicates()

