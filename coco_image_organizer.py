import json
import os
import shutil
from pathlib import Path

def organize_coco_images(json_file_path, search_folders, output_folder):
    """
    Find and copy images from COCO JSON file to a new organized folder.
    
    Args:
        json_file_path (str): Path to the combined COCO JSON file
        search_folders (list): List of folder paths to search for images
        output_folder (str): Path to the output folder where images will be copied
    """
    # Load the COCO JSON file
    try:
        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_path}")
    
    # Convert search folders to Path objects
    search_paths = [Path(folder) for folder in search_folders]
    
    # Get all image filenames from the COCO data
    images = coco_data.get('images', [])
    if not images:
        print("No images found in the COCO JSON file")
        return
    
    print(f"Found {len(images)} images to process")
    
    # Track statistics
    found_count = 0
    not_found_count = 0
    copied_count = 0
    already_exists_count = 0
    
    # Process each image
    for i, image_info in enumerate(images):
        filename = image_info['file_name']
        print(f"Processing ({i+1}/{len(images)}): {filename}")
        
        # Search for the image in all folders
        image_found = False
        source_path = None
        
        for search_path in search_paths:
            potential_path = search_path / filename
            if potential_path.exists():
                source_path = potential_path
                image_found = True
                print(f"  Found in: {search_path}")
                break
        
        if not image_found:
            print(f"  ❌ Not found in any search folder")
            not_found_count += 1
            continue
        
        found_count += 1
        
        # Copy the image to the output folder
        destination_path = output_path / filename
        
        if destination_path.exists():
            print(f"  ⚠️  Already exists in output folder")
            already_exists_count += 1
            continue
        
        try:
            shutil.copy2(source_path, destination_path)
            print(f"  ✅ Copied successfully")
            copied_count += 1
        except Exception as e:
            print(f"  ❌ Error copying: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"{'='*50}")
    print(f"Total images in JSON: {len(images)}")
    print(f"Images found: {found_count}")
    print(f"Images not found: {not_found_count}")
    print(f"Images copied: {copied_count}")
    print(f"Images already existed: {already_exists_count}")
    print(f"Output folder: {output_path}")
    
    # List not found images if any
    if not_found_count > 0:
        print(f"\n⚠️  {not_found_count} images were not found in any search folder")

def main():
    print("COCO Image Organizer")
    print("===================")
    
    # Get COCO JSON file path
    json_file = input("Enter path to the combined COCO JSON file: ").strip()
    if not json_file:
        print("No JSON file provided. Exiting.")
        return
    
    if not Path(json_file).exists():
        print(f"JSON file not found: {json_file}")
        return
    
    # Get search folders
    print("\nEnter the folders to search for images (one per line).")
    print("Press Enter twice when done:")
    
    search_folders = []
    while True:
        folder = input().strip()
        if not folder:
            break
        if Path(folder).exists():
            search_folders.append(folder)
            print(f"  Added: {folder}")
        else:
            print(f"  ⚠️  Folder not found: {folder}")
    
    if not search_folders:
        print("No valid search folders provided. Exiting.")
        return
    
    # Get output folder
    output_folder = input("\nEnter output folder path: ").strip()
    if not output_folder:
        print("No output folder provided. Exiting.")
        return
    
    # Confirm before proceeding
    print(f"\n{'='*50}")
    print("CONFIGURATION:")
    print(f"{'='*50}")
    print(f"JSON file: {json_file}")
    print(f"Search folders ({len(search_folders)}):")
    for folder in search_folders:
        print(f"  - {folder}")
    print(f"Output folder: {output_folder}")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return
    
    try:
        organize_coco_images(json_file, search_folders, output_folder)
    except Exception as e:
        print(f"Error: {e}")

# Alternative: Direct usage
def organize_images_directly(json_file, search_folders, output_folder):
    """
    Direct function to organize images - useful if you want to modify the paths in the script
    
    Example usage:
    json_file = "path/to/combined.json"
    search_folders = [
        "folder1",
        "folder2", 
        "folder3",
        # ... up to 10 folders
    ]
    output_folder = "organized_images"
    organize_images_directly(json_file, search_folders, output_folder)
    """
    organize_coco_images(json_file, search_folders, output_folder)

if __name__ == "__main__":
    main()
