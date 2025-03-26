"""
Utility functions for the Squarespace Thread Art app
"""
import os
import shutil
from pathlib import Path

def ensure_demo_images():
    """
    Ensures that the demo images are available in the squarespace/demo_images directory
    """
    # Get the parent directory of the squarespace folder
    app_dir = Path(__file__).parent
    root_dir = app_dir.parent
    
    # Create demo_images directory if it doesn't exist
    demo_dir = app_dir / "demo_images"
    if not demo_dir.exists():
        demo_dir.mkdir(parents=True)
    
    # Check for tiger.jpg and copy it if needed
    tiger_source = root_dir / "images" / "tiger.jpg"
    tiger_dest = demo_dir / "tiger.jpg"
    
    if tiger_source.exists() and not tiger_dest.exists():
        shutil.copy(tiger_source, tiger_dest)
        print(f"Copied tiger.jpg to {tiger_dest}")
    elif not tiger_source.exists():
        print(f"Warning: Source image {tiger_source} not found")

def cleanup_temp_files(temp_dir):
    """
    Removes temporary files created during thread art generation
    """
    try:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
        print(f"Cleaned up temporary files in {temp_dir}")
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")
