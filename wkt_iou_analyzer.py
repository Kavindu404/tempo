import pandas as pd
from shapely import wkt
from shapely.geometry import shape
import warnings
warnings.filterwarnings('ignore')

def calculate_iou(geom1_wkt, geom2_wkt):
    """Calculate Intersection over Union between two WKT geometries."""
    try:
        if pd.isna(geom1_wkt) or pd.isna(geom2_wkt):
            return None
        
        geom1 = wkt.loads(str(geom1_wkt))
        geom2 = wkt.loads(str(geom2_wkt))
        
        if not geom1.is_valid or not geom2.is_valid:
            return None
        
        intersection = geom1.intersection(geom2).area
        union = geom1.union(geom2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except:
        return None

def analyze_wkt_file(input_file):
    """Analyze Excel file with WKT columns and generate IoU statistics."""
    
    # Read the Excel file
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file)
    
    # Calculate IoU for building geometries
    print("Calculating building IoU values...")
    df['iou_ecopia_precisely_bldg'] = df.apply(
        lambda row: calculate_iou(row['Ecopia_BLDG_WKT'], row['Precisely_Bldg_WKT']), axis=1
    )
    df['iou_ecopia_lightbox_bldg'] = df.apply(
        lambda row: calculate_iou(row['Ecopia_BLDG_WKT'], row['lightbox_bldg_wkt']), axis=1
    )
    df['iou_precisely_lightbox_bldg'] = df.apply(
        lambda row: calculate_iou(row['Precisely_Bldg_WKT'], row['lightbox_bldg_wkt']), axis=1
    )
    
    # Calculate IoU for parcel geometries
    print("Calculating parcel IoU values...")
    df['iou_ecopia_precisely_prcl'] = df.apply(
        lambda row: calculate_iou(row['Ecopia_Prcl_WKT'], row['Precisely_Prcl_WKT']), axis=1
    )
    df['iou_ecopia_lightbox_prcl'] = df.apply(
        lambda row: calculate_iou(row['Ecopia_Prcl_WKT'], row['lightbox_prcl_wkt']), axis=1
    )
    df['iou_precisely_lightbox_prcl'] = df.apply(
        lambda row: calculate_iou(row['Precisely_Prcl_WKT'], row['lightbox_prcl_wkt']), axis=1
    )
    
    # Save analyzed Excel file
    output_file = input_file.rsplit('.', 1)[0] + '_analyzed.xlsx'
    print(f"Saving analyzed file to {output_file}...")
    df.to_excel(output_file, index=False)
    
    # Generate statistics
    print("Generating statistics...")
    stats = []
    indices = {}
    
    # Building IoU < 0.5 statistics
    idx = df[df['iou_ecopia_precisely_bldg'] < 0.5].index.tolist()
    stats.append(f"Rows where Ecopia-Precisely building IoU < 0.5: {len(idx)}")
    indices['ecopia_precisely_bldg_low'] = idx
    
    idx = df[df['iou_ecopia_lightbox_bldg'] < 0.5].index.tolist()
    stats.append(f"Rows where Ecopia-Lightbox building IoU < 0.5: {len(idx)}")
    indices['ecopia_lightbox_bldg_low'] = idx
    
    idx = df[df['iou_precisely_lightbox_bldg'] < 0.5].index.tolist()
    stats.append(f"Rows where Precisely-Lightbox building IoU < 0.5: {len(idx)}")
    indices['precisely_lightbox_bldg_low'] = idx
    
    # Parcel IoU < 0.5 statistics
    idx = df[df['iou_ecopia_precisely_prcl'] < 0.5].index.tolist()
    stats.append(f"Rows where Ecopia-Precisely parcel IoU < 0.5: {len(idx)}")
    indices['ecopia_precisely_prcl_low'] = idx
    
    idx = df[df['iou_ecopia_lightbox_prcl'] < 0.5].index.tolist()
    stats.append(f"Rows where Ecopia-Lightbox parcel IoU < 0.5: {len(idx)}")
    indices['ecopia_lightbox_prcl_low'] = idx
    
    idx = df[df['iou_precisely_lightbox_prcl'] < 0.5].index.tolist()
    stats.append(f"Rows where Precisely-Lightbox parcel IoU < 0.5: {len(idx)}")
    indices['precisely_lightbox_prcl_low'] = idx
    
    # Rows where Ecopia and Precisely have building WKTs but Lightbox doesn't
    idx = df[
        df['Ecopia_BLDG_WKT'].notna() & 
        df['Precisely_Bldg_WKT'].notna() & 
        df['lightbox_bldg_wkt'].isna()
    ].index.tolist()
    stats.append(f"Rows where Ecopia and Precisely have building WKTs but Lightbox doesn't: {len(idx)}")
    indices['ecopia_precisely_have_bldg_lightbox_no'] = idx
    
    # Rows where Ecopia and Precisely have parcel WKTs but Lightbox doesn't
    idx = df[
        df['Ecopia_Prcl_WKT'].notna() & 
        df['Precisely_Prcl_WKT'].notna() & 
        df['lightbox_prcl_wkt'].isna()
    ].index.tolist()
    stats.append(f"Rows where Ecopia and Precisely have parcel WKTs but Lightbox doesn't: {len(idx)}")
    indices['ecopia_precisely_have_prcl_lightbox_no'] = idx
    
    # Rows where none have WKTs for both building and parcel
    idx = df[
        df['Ecopia_BLDG_WKT'].isna() & 
        df['Precisely_Bldg_WKT'].isna() & 
        df['lightbox_bldg_wkt'].isna() &
        df['Ecopia_Prcl_WKT'].isna() & 
        df['Precisely_Prcl_WKT'].isna() & 
        df['lightbox_prcl_wkt'].isna()
    ].index.tolist()
    stats.append(f"Rows where none have WKTs for both building and parcel: {len(idx)}")
    indices['none_have_both'] = idx
    
    # Rows where Ecopia vs Precisely IoU > 0.5 but vs Lightbox IoU < 0.5 for both building and parcel
    idx = df[
        (df['iou_ecopia_precisely_bldg'] > 0.5) & 
        (df['iou_ecopia_lightbox_bldg'] < 0.5) &
        (df['iou_ecopia_precisely_prcl'] > 0.5) & 
        (df['iou_ecopia_lightbox_prcl'] < 0.5)
    ].index.tolist()
    stats.append(f"Rows where Ecopia-Precisely IoU > 0.5 but vs Lightbox IoU < 0.5 for both building and parcel: {len(idx)}")
    indices['good_ecopia_precisely_bad_lightbox'] = idx
    
    # Save statistics to text file
    stats_file = input_file.rsplit('.', 1)[0] + '_analyzed.txt'
    print(f"Saving statistics to {stats_file}...")
    
    with open(stats_file, 'w') as f:
        # Write statistics first
        f.write("=" * 80 + "\n")
        f.write("STATISTICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        for stat in stats:
            f.write(stat + "\n")
        
        # Write indices
        f.write("\n" + "=" * 80 + "\n")
        f.write("ROW INDICES\n")
        f.write("=" * 80 + "\n\n")
        
        for key, idx_list in indices.items():
            f.write(f"\n{key}:\n")
            f.write(f"{idx_list}\n")
    
    print("Analysis complete!")
    print(f"Output files: {output_file}, {stats_file}")

if __name__ == "__main__":
    # Replace with your input file path
    input_file = "your_file.xlsx"
    analyze_wkt_file(input_file)
