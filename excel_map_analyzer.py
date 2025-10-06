import pandas as pd
import os
import shutil
from pathlib import Path


def analyze_excel_and_copy_maps(
    excel_path,
    maps_source_dir,
    output_dir="filtered_maps",
    address_column="Full Address"
):
    """
    Analyze Excel file for data quality issues and copy corresponding map files.
    
    Parameters:
    -----------
    excel_path : str
        Path to the Excel file
    maps_source_dir : str
        Directory containing the HTML map files
    output_dir : str
        Directory to copy filtered maps to (default: "filtered_maps")
    address_column : str
        Name of the column containing full addresses (default: "Full Address")
    
    Returns:
    --------
    dict : Analysis results and statistics
    """
    
    # Load the Excel file
    print(f"Loading Excel file: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f"Total rows loaded: {len(df)}")
    
    # Analysis results dictionary
    results = {}
    
    # 1. Count rows where Google_In_Ecopia_Prcl is not 1
    google_not_1 = df[df['Google_In_Ecopia_Prcl'] != 1]
    results['google_not_1_count'] = len(google_not_1)
    print(f"\nGoogle_In_Ecopia_Prcl not 1: {results['google_not_1_count']}")
    
    # 2. Count rows where Google_In_Ecopia_Bldg is not 1
    bldg_not_1 = df[df['Google_In_Ecopia_Bldg'] != 1]
    results['bldg_not_1_count'] = len(bldg_not_1)
    print(f"Google_In_Ecopia_Bldg not 1: {results['bldg_not_1_count']}")
    
    # 3. Count rows where Precisely_In_Ecopia is not 1
    precisely_not_1 = df[df['Precisely_In_Ecopia'] != 1]
    results['precisely_not_1_count'] = len(precisely_not_1)
    print(f"Precisely_In_Ecopia not 1: {results['precisely_not_1_count']}")
    
    # 4. Among those, how many are 0s when Google_Location_Type is ROOFTOP
    google_0_rooftop = df[
        (df['Google_In_Ecopia_Prcl'] == 0) & 
        (df['Google_Location_Type'] == 'ROOFTOP')
    ]
    results['google_0_rooftop_count'] = len(google_0_rooftop)
    print(f"\nGoogle_In_Ecopia_Prcl = 0 with ROOFTOP: {results['google_0_rooftop_count']}")
    
    bldg_0_rooftop = df[
        (df['Google_In_Ecopia_Bldg'] == 0) & 
        (df['Google_Location_Type'] == 'ROOFTOP')
    ]
    results['bldg_0_rooftop_count'] = len(bldg_0_rooftop)
    print(f"Google_In_Ecopia_Bldg = 0 with ROOFTOP: {results['bldg_0_rooftop_count']}")
    
    precisely_0_rooftop = df[
        (df['Precisely_In_Ecopia'] == 0) & 
        (df['Google_Location_Type'] == 'ROOFTOP')
    ]
    results['precisely_0_rooftop_count'] = len(precisely_0_rooftop)
    print(f"Precisely_In_Ecopia = 0 with ROOFTOP: {results['precisely_0_rooftop_count']}")
    
    # 5. Ecopia_Confident_Level not equal to 9
    conf_not_9 = df[df['Ecopia_Confident_Level'] != 9]
    results['conf_not_9_count'] = len(conf_not_9)
    print(f"\nEcopia_Confident_Level != 9: {results['conf_not_9_count']}")
    
    # 6. Ecopia_Geo_Precision not equal to 9
    geo_not_9 = df[df['Ecopia_Geo_Precision'] != 9]
    results['geo_not_9_count'] = len(geo_not_9)
    print(f"Ecopia_Geo_Precision != 9: {results['geo_not_9_count']}")
    
    # 7. Intersection of both (not 9 in both columns)
    both_not_9 = df[
        (df['Ecopia_Confident_Level'] != 9) & 
        (df['Ecopia_Geo_Precision'] != 9)
    ]
    results['both_not_9_count'] = len(both_not_9)
    print(f"Both Ecopia metrics != 9: {results['both_not_9_count']}")
    
    # 8. Combine all filtered rows (union of all conditions)
    filtered_df = pd.concat([
        google_not_1,
        bldg_not_1,
        precisely_not_1,
        conf_not_9,
        geo_not_9
    ]).drop_duplicates()
    
    print(f"\nTotal unique rows matching any criteria: {len(filtered_df)}")
    results['total_filtered_rows'] = len(filtered_df)
    
    # 9. Copy corresponding map files
    print(f"\nCopying map files to: {output_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    maps_source_path = Path(maps_source_dir)
    copied_count = 0
    not_found_count = 0
    not_found_addresses = []
    
    for idx, row in filtered_df.iterrows():
        address = row[address_column]
        # Map filename format: {idx}_{Full Address}.html
        map_filename = f"{idx}_{address}.html"
        source_path = maps_source_path / map_filename
        
        if source_path.exists():
            destination_path = Path(output_dir) / map_filename
            shutil.copy2(source_path, destination_path)
            copied_count += 1
        else:
            not_found_count += 1
            not_found_addresses.append((idx, address))
    
    print(f"Maps copied: {copied_count}")
    print(f"Maps not found: {not_found_count}")
    
    results['maps_copied'] = copied_count
    results['maps_not_found'] = not_found_count
    results['not_found_addresses'] = not_found_addresses
    
    # Save filtered addresses to a CSV for reference
    filtered_addresses_path = Path(output_dir) / "filtered_addresses.csv"
    filtered_df[[address_column]].to_csv(filtered_addresses_path, index=True)
    print(f"\nFiltered addresses saved to: {filtered_addresses_path}")
    
    # Save detailed statistics
    stats_path = Path(output_dir) / "analysis_statistics.txt"
    with open(stats_path, 'w') as f:
        f.write("=== Excel Analysis Statistics ===\n\n")
        f.write(f"Total rows: {len(df)}\n\n")
        f.write(f"Google_In_Ecopia_Prcl != 1: {results['google_not_1_count']}\n")
        f.write(f"Google_In_Ecopia_Bldg != 1: {results['bldg_not_1_count']}\n")
        f.write(f"Precisely_In_Ecopia != 1: {results['precisely_not_1_count']}\n\n")
        f.write(f"Google_In_Ecopia_Prcl = 0 with ROOFTOP: {results['google_0_rooftop_count']}\n")
        f.write(f"Google_In_Ecopia_Bldg = 0 with ROOFTOP: {results['bldg_0_rooftop_count']}\n")
        f.write(f"Precisely_In_Ecopia = 0 with ROOFTOP: {results['precisely_0_rooftop_count']}\n\n")
        f.write(f"Ecopia_Confident_Level != 9: {results['conf_not_9_count']}\n")
        f.write(f"Ecopia_Geo_Precision != 9: {results['geo_not_9_count']}\n")
        f.write(f"Both Ecopia metrics != 9: {results['both_not_9_count']}\n\n")
        f.write(f"Total filtered rows: {results['total_filtered_rows']}\n")
        f.write(f"Maps copied: {results['maps_copied']}\n")
        f.write(f"Maps not found: {results['maps_not_found']}\n")
    
    print(f"Statistics saved to: {stats_path}")
    
    return results


# Example usage:
if __name__ == "__main__":
    results = analyze_excel_and_copy_maps(
        excel_path="your_data.xlsx",
        maps_source_dir="./maps",
        output_dir="filtered_maps",
        address_column="Full Address"
    )
    
    print("\n=== Analysis Complete ===")
    print(f"Results: {results}")
