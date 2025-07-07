import json
import os
from pathlib import Path

def combine_coco_files(file_paths, output_filename=None):
    """
    Combine multiple COCO annotated JSON files into a single file.
    
    Args:
        file_paths (list): List of paths to COCO JSON files
        output_filename (str): Optional custom output filename
    """
    # Convert to Path objects and validate
    json_files = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        if not path.suffix.lower() == '.json':
            print(f"Warning: Not a JSON file: {file_path}")
            continue
        json_files.append(path)
    
    if not json_files:
        print("No valid JSON files provided")
        return
    
    print(f"Processing {len(json_files)} JSON files")
    
    # Initialize combined structure
    combined_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Track category mappings and used IDs
    category_map = {}  # old_id -> new_id
    used_category_ids = set()
    next_category_id = 1
    
    # Track ID mappings
    image_id_map = {}  # (file_index, old_id) -> new_id
    next_image_id = 1
    next_annotation_id = 1
    
    # Process each file
    for file_idx, json_file in enumerate(json_files):
        print(f"Processing: {json_file.name}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Copy info and licenses from first file
        if file_idx == 0:
            if 'info' in data:
                combined_data['info'] = data['info']
            if 'licenses' in data:
                combined_data['licenses'] = data['licenses']
        
        # Process categories
        file_category_map = {}
        for category in data.get('categories', []):
            old_cat_id = category['id']
            
            # Check if this category already exists (by name)
            existing_cat_id = None
            for existing_cat in combined_data['categories']:
                if existing_cat['name'] == category['name']:
                    existing_cat_id = existing_cat['id']
                    break
            
            if existing_cat_id:
                file_category_map[old_cat_id] = existing_cat_id
            else:
                # Find next available category ID
                while next_category_id in used_category_ids:
                    next_category_id += 1
                
                new_cat_id = next_category_id
                used_category_ids.add(new_cat_id)
                file_category_map[old_cat_id] = new_cat_id
                
                # Add category with new ID
                new_category = category.copy()
                new_category['id'] = new_cat_id
                combined_data['categories'].append(new_category)
                
                next_category_id += 1
        
        # Process images
        file_image_map = {}
        for image in data.get('images', []):
            old_img_id = image['id']
            new_img_id = next_image_id
            
            file_image_map[old_img_id] = new_img_id
            image_id_map[(file_idx, old_img_id)] = new_img_id
            
            # Add image with new ID
            new_image = image.copy()
            new_image['id'] = new_img_id
            combined_data['images'].append(new_image)
            
            next_image_id += 1
        
        # Process annotations
        for annotation in data.get('annotations', []):
            new_annotation = annotation.copy()
            new_annotation['id'] = next_annotation_id
            new_annotation['image_id'] = file_image_map[annotation['image_id']]
            new_annotation['category_id'] = file_category_map[annotation['category_id']]
            
            combined_data['annotations'].append(new_annotation)
            next_annotation_id += 1
    
    # Create output filename
    if output_filename is None:
        # Use the directory of the first file and add _combined
        first_file = json_files[0]
        output_filename = f"{first_file.stem}_combined.json"
    
    output_path = json_files[0].parent / output_filename
    
    # Save combined file
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"\nCombined file saved as: {output_path}")
    print(f"Total images: {len(combined_data['images'])}")
    print(f"Total annotations: {len(combined_data['annotations'])}")
    print(f"Total categories: {len(combined_data['categories'])}")

def main():
    # Get file paths from user
    print("Enter the paths to your COCO JSON files (one per line).")
    print("Press Enter twice when done:")
    
    file_paths = []
    while True:
        file_path = input().strip()
        if not file_path:
            break
        file_paths.append(file_path)
    
    if not file_paths:
        print("No files provided. Exiting.")
        return
    
    # Optional custom output filename
    custom_name = input("Enter custom output filename (or press Enter for auto-generated): ").strip()
    output_filename = custom_name if custom_name else None
    
    try:
        combine_coco_files(file_paths, output_filename)
    except Exception as e:
        print(f"Error: {e}")

# Alternative: Direct usage with file paths
def combine_files_directly(file_paths, output_name=None):
    """
    Direct function to combine files - useful if you want to modify the file paths in the script
    
    Example usage:
    file_paths = [
        "path/to/file1.json",
        "path/to/file2.json", 
        "path/to/file3.json"
    ]
    combine_files_directly(file_paths)
    """
    combine_coco_files(file_paths, output_name)

if __name__ == "__main__":
    main()
