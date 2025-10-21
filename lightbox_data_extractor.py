import pandas as pd
import json
from pathlib import Path
from typing import List, Optional, Dict, Any


def load_json_file(filepath: Path) -> Optional[Dict[Any, Any]]:
    """Load JSON file if it exists, return None otherwise."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON file {filepath}")
        return None


def extract_geocoded_data(data: Optional[Dict]) -> tuple:
    """Extract confidence score, latitude, and longitude from geocoded address."""
    if not data:
        return None, None, None
    
    try:
        geocoded = data.get('Geocoded Address', {})
        addresses = geocoded.get('addresses', [])
        
        if addresses:
            first_addr = addresses[0]
            # Confidence score
            confidence = first_addr.get('$metadata', {}).get('geocode', {}).get('confidence', {}).get('score')
            
            # Location
            location = first_addr.get('location', {})
            rep_point = location.get('representativePoint', {})
            latitude = rep_point.get('latitude')
            longitude = rep_point.get('longitude')
            
            return confidence, latitude, longitude
    except (KeyError, IndexError, AttributeError):
        pass
    
    return None, None, None


def extract_parcel_data(data: Optional[Dict]) -> tuple:
    """Extract WKT geometries and coordinates from parcel data."""
    if not data:
        return None, [], []
    
    try:
        parcel_data = data.get('Parcel Data', {})
        parcels = parcel_data.get('parcels', [])
        
        if not parcels:
            return None, [], []
        
        wkts = []
        lats = []
        lons = []
        
        for parcel in parcels:
            # Get WKT
            geometry = parcel.get('location', {}).get('geometry', {})
            wkt = geometry.get('wkt')
            if wkt:
                wkts.append(wkt)
            
            # Get coordinates
            rep_point = parcel.get('representativePoint', {})
            lat = rep_point.get('latitude')
            lon = rep_point.get('longitude')
            if lat is not None:
                lats.append(lat)
            if lon is not None:
                lons.append(lon)
        
        # Combine WKTs into a GEOMETRYCOLLECTION if multiple
        combined_wkt = None
        if len(wkts) == 1:
            combined_wkt = wkts[0]
        elif len(wkts) > 1:
            combined_wkt = f"GEOMETRYCOLLECTION({', '.join(wkts)})"
        
        return combined_wkt, lats, lons
    except (KeyError, AttributeError):
        pass
    
    return None, [], []


def extract_structure_data(data: Optional[Dict]) -> tuple:
    """Extract WKT geometries and coordinates from structure data."""
    if not data:
        return None, [], []
    
    try:
        structures_data = data.get('Structures on Parcel', {})
        structures = structures_data.get('structures', [])
        
        if not structures:
            return None, [], []
        
        wkts = []
        lats = []
        lons = []
        
        for structure in structures:
            # Get WKT
            geometry = structure.get('location', {}).get('geometry', {})
            wkt = geometry.get('wkt')
            if wkt:
                wkts.append(wkt)
            
            # Get coordinates
            rep_point = structure.get('location', {}).get('representativePoint', {})
            lat = rep_point.get('latitude')
            lon = rep_point.get('longitude')
            if lat is not None:
                lats.append(lat)
            if lon is not None:
                lons.append(lon)
        
        # Combine WKTs into a GEOMETRYCOLLECTION if multiple
        combined_wkt = None
        if len(wkts) == 1:
            combined_wkt = wkts[0]
        elif len(wkts) > 1:
            combined_wkt = f"GEOMETRYCOLLECTION({', '.join(wkts)})"
        
        return combined_wkt, lats, lons
    except (KeyError, AttributeError):
        pass
    
    return None, [], []


def process_excel_with_lightbox_data(excel_path: str, json_dir: str):
    """
    Process Excel file and add Lightbox data columns.
    
    Args:
        excel_path: Path to the Excel file
        json_dir: Directory containing the JSON files
    """
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Initialize new columns
    df['lightbox_address_confidence'] = None
    df['lightbox_addr_lat'] = None
    df['lightbox_addr_lon'] = None
    df['lightbox_prcl_wkt'] = None
    df['lightbox_prcl_lat'] = None
    df['lightbox_prcl_lon'] = None
    df['lightbox_bldg_wkt'] = None
    df['lightbox_bldg_lat'] = None
    df['lightbox_bldg_lon'] = None
    
    json_dir_path = Path(json_dir)
    
    # Process each row
    for idx in df.index:
        json_file = json_dir_path / f"{idx}_address.json"
        
        # Load JSON data
        data = load_json_file(json_file)
        
        if data:
            # Extract geocoded address data
            confidence, addr_lat, addr_lon = extract_geocoded_data(data)
            df.at[idx, 'lightbox_address_confidence'] = confidence
            df.at[idx, 'lightbox_addr_lat'] = addr_lat
            df.at[idx, 'lightbox_addr_lon'] = addr_lon
            
            # Extract parcel data
            prcl_wkt, prcl_lats, prcl_lons = extract_parcel_data(data)
            df.at[idx, 'lightbox_prcl_wkt'] = prcl_wkt
            df.at[idx, 'lightbox_prcl_lat'] = prcl_lats if prcl_lats else None
            df.at[idx, 'lightbox_prcl_lon'] = prcl_lons if prcl_lons else None
            
            # Extract structure data
            bldg_wkt, bldg_lats, bldg_lons = extract_structure_data(data)
            df.at[idx, 'lightbox_bldg_wkt'] = bldg_wkt
            df.at[idx, 'lightbox_bldg_lat'] = bldg_lats if bldg_lats else None
            df.at[idx, 'lightbox_bldg_lon'] = bldg_lons if bldg_lons else None
    
    # Save to new file
    excel_path_obj = Path(excel_path)
    output_path = excel_path_obj.parent / f"{excel_path_obj.stem}_lightbox{excel_path_obj.suffix}"
    df.to_excel(output_path, index=False)
    
    print(f"Processing complete! Saved to: {output_path}")
    return output_path


# Example usage
if __name__ == "__main__":
    # Update these paths
    excel_file = "your_file.xlsx"
    json_directory = "path/to/json/directory"
    
    process_excel_with_lightbox_data(excel_file, json_directory)
