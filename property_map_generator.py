import pandas as pd
import folium
from folium import plugins
import json
import ast
import os
from typing import List, Dict, Any, Union


def create_property_map(excel_file_path: str, output_dir: str, map_filename: str = "property_map.html"):
    """
    Create an interactive Folium map from Excel property data.
    
    Parameters:
    -----------
    excel_file_path : str
        Path to the Excel file containing property data
    output_dir : str
        Directory where the map HTML file will be saved
    map_filename : str, optional
        Name of the output HTML file (default: "property_map.html")
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read Excel file
    df = pd.read_excel(excel_file_path)
    
    # Initialize the map (centered on first address or default location)
    if not df.empty and pd.notna(df.iloc[0]['Addr_latitude']):
        addr_lats = safe_eval(df.iloc[0]['Addr_latitude'])
        addr_lons = safe_eval(df.iloc[0]['Addr_longitude'])
        if addr_lats and addr_lons:
            center_lat = float(addr_lats[0])
            center_lon = float(addr_lons[0])
        else:
            center_lat, center_lon = 39.8283, -98.5795  # Center of US as fallback
    else:
        center_lat, center_lon = 39.8283, -98.5795  # Center of US as fallback
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add different tile layers for better visualization
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    
    # Create feature groups for different layers
    address_group = folium.FeatureGroup(name='Address Markers', show=True)
    parcel_group = folium.FeatureGroup(name='Parcels', show=True)
    building_group = folium.FeatureGroup(name='Buildings', show=True)
    parcel_points_group = folium.FeatureGroup(name='Parcel Center Points', show=False)
    building_points_group = folium.FeatureGroup(name='Building Center Points', show=False)
    
    # Color schemes for different elements
    colors = {
        'address': 'red',
        'parcel': 'blue',
        'building': 'green',
        'parcel_point': 'darkblue',
        'building_point': 'darkgreen'
    }
    
    def safe_eval(value):
        """Safely evaluate string representations of lists"""
        if pd.isna(value) or value == '':
            return []
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
        return value if isinstance(value, list) else [value]
    
    def create_popup_content(row, index):
        """Create HTML content for popup information card"""
        # Handle Full Address as a list
        full_addresses = safe_eval(row.get('Full Address', []))
        address_text = full_addresses[0] if full_addresses else 'Unknown Address'
        
        popup_html = f"""
        <div style="width: 300px; font-family: Arial, sans-serif;">
            <h3 style="color: #333; margin-bottom: 10px; border-bottom: 2px solid #ddd; padding-bottom: 5px;">
                {address_text}
            </h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td style="padding: 3px; font-weight: bold;">PreciselyID:</td>
                    <td style="padding: 3px;">{row.get('PreciselyID', 'N/A')}</td></tr>
                <tr><td style="padding: 3px; font-weight: bold;">FIPS:</td>
                    <td style="padding: 3px;">{row.get('FIPS', 'N/A')}</td></tr>
                <tr><td style="padding: 3px; font-weight: bold;">GeographyID:</td>
                    <td style="padding: 3px;">{row.get('GeographyID', 'N/A')}</td></tr>
                <tr><td style="padding: 3px; font-weight: bold;">Building Elevation:</td>
                    <td style="padding: 3px;">{row.get('BLDG_Elevation', 'N/A')}</td></tr>
                <tr><td style="padding: 3px; font-weight: bold;">Max Elevation:</td>
                    <td style="padding: 3px;">{row.get('MaximumElevation', 'N/A')}</td></tr>
                <tr><td style="padding: 3px; font-weight: bold;">Min Elevation:</td>
                    <td style="padding: 3px;">{row.get('MinimumElevation', 'N/A')}</td></tr>
            </table>
        </div>
        """
        return popup_html
    
    def add_geometry_to_map(coordinates, geometry_type, feature_group, color, popup_content):
        """Add polygon or multipolygon geometry to the map"""
        try:
            if geometry_type.lower() == 'polygon':
                # For polygon, coordinates should be [[[lon, lat], [lon, lat], ...]]
                if coordinates and len(coordinates) > 0:
                    # Convert from [lon, lat] to [lat, lon] for folium
                    polygon_coords = [[coord[1], coord[0]] for coord in coordinates[0]]
                    folium.Polygon(
                        locations=polygon_coords,
                        color=color,
                        weight=2,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.3,
                        popup=folium.Popup(popup_content, max_width=350)
                    ).add_to(feature_group)
                    
            elif geometry_type.lower() == 'multipolygon':
                # For multipolygon, coordinates should be [[[[lon, lat], ...]], [[[lon, lat], ...]]]
                if coordinates and len(coordinates) > 0:
                    for polygon in coordinates:
                        if polygon and len(polygon) > 0:
                            # Convert from [lon, lat] to [lat, lon] for folium
                            polygon_coords = [[coord[1], coord[0]] for coord in polygon[0]]
                            folium.Polygon(
                                locations=polygon_coords,
                                color=color,
                                weight=2,
                                fill=True,
                                fillColor=color,
                                fillOpacity=0.3,
                                popup=folium.Popup(popup_content, max_width=350)
                            ).add_to(feature_group)
        except Exception as e:
            print(f"Error adding geometry: {e}")
    
    # Process each row in the dataframe
    for index, row in df.iterrows():
        popup_content = create_popup_content(row, index)
        
        # Process address markers (can be multiple addresses per row)
        addr_lats = safe_eval(row.get('Addr_latitude', []))
        addr_lons = safe_eval(row.get('Addr_longitude', []))
        full_addresses = safe_eval(row.get('Full Address', []))
        
        # Ensure we have the same number of addresses, lats, and lons
        max_addr_count = max(len(addr_lats), len(addr_lons), len(full_addresses))
        
        for i in range(max_addr_count):
            # Get address info, using last available if lists are different lengths
            addr_lat = float(addr_lats[min(i, len(addr_lats)-1)]) if addr_lats else None
            addr_lon = float(addr_lons[min(i, len(addr_lons)-1)]) if addr_lons else None
            address_text = full_addresses[min(i, len(full_addresses)-1)] if full_addresses else f"Address {i+1}"
            
            if addr_lat is not None and addr_lon is not None:
                folium.Marker(
                    location=[addr_lat, addr_lon],
                    popup=folium.Popup(popup_content, max_width=350),
                    tooltip=f"Address: {address_text}",
                    icon=folium.Icon(color=colors['address'], icon='home', prefix='fa')
                ).add_to(address_group)
        
        # Process parcels
        prcl_lats = safe_eval(row.get('PRCL_latitude', []))
        prcl_lons = safe_eval(row.get('PRCL_longitude', []))
        prcl_geom_types = safe_eval(row.get('PRCL_Geometry_type', []))
        prcl_coordinates = safe_eval(row.get('PRCL_Geometry_coordinates', []))
        
        # Add parcel geometries
        for i in range(len(prcl_geom_types)):
            if i < len(prcl_coordinates):
                add_geometry_to_map(
                    prcl_coordinates[i], 
                    prcl_geom_types[i], 
                    parcel_group, 
                    colors['parcel'], 
                    popup_content
                )
        
        # Add parcel center points
        for i in range(len(prcl_lats)):
            if i < len(prcl_lons):
                try:
                    lat_float = float(prcl_lats[i])
                    lon_float = float(prcl_lons[i])
                    folium.CircleMarker(
                        location=[lat_float, lon_float],
                        radius=5,
                        popup=folium.Popup(f"Parcel {i+1}<br>" + popup_content, max_width=350),
                        tooltip=f"Parcel {i+1}",
                        color=colors['parcel_point'],
                        fill=True,
                        fillColor=colors['parcel_point']
                    ).add_to(parcel_points_group)
                except (ValueError, TypeError):
                    continue
        
        # Process buildings
        bldg_lats = safe_eval(row.get('BLDG_lattitude', []))  # Note: using your spelling
        bldg_lons = safe_eval(row.get('BLDG_Longitude', []))
        bldg_geom_types = safe_eval(row.get('BLDG_Geometry_type', []))
        bldg_coordinates = safe_eval(row.get('BLDG_Geometry_Coordinates', []))
        
        # Add building geometries
        for i in range(len(bldg_geom_types)):
            if i < len(bldg_coordinates):
                add_geometry_to_map(
                    bldg_coordinates[i], 
                    bldg_geom_types[i], 
                    building_group, 
                    colors['building'], 
                    popup_content
                )
        
        # Add building center points
        for i in range(len(bldg_lats)):
            if i < len(bldg_lons):
                try:
                    lat_float = float(bldg_lats[i])
                    lon_float = float(bldg_lons[i])
                    folium.CircleMarker(
                        location=[lat_float, lon_float],
                        radius=5,
                        popup=folium.Popup(f"Building {i+1}<br>" + popup_content, max_width=350),
                        tooltip=f"Building {i+1}",
                        color=colors['building_point'],
                        fill=True,
                        fillColor=colors['building_point']
                    ).add_to(building_points_group)
                except (ValueError, TypeError):
                    continue
    
    # Add all feature groups to the map
    address_group.add_to(m)
    parcel_group.add_to(m)
    building_group.add_to(m)
    parcel_points_group.add_to(m)
    building_points_group.add_to(m)
    
    # Add layer control for filtering
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Legend</h4>
    <p><i class="fa fa-home" style="color:red"></i> Address Markers</p>
    <p><span style="color:blue">■</span> Parcels</p>
    <p><span style="color:green">■</span> Buildings</p>
    <p><span style="color:darkblue">●</span> Parcel Centers</p>
    <p><span style="color:darkgreen">●</span> Building Centers</p>
    <p><small>Use layer control to toggle visibility</small></p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add fullscreen plugin
    plugins.Fullscreen().add_to(m)
    
    # Save the map
    output_path = os.path.join(output_dir, map_filename)
    m.save(output_path)
    
    print(f"Map saved successfully to: {output_path}")
    print(f"Processed {len(df)} addresses")
    return output_path


# Example usage:
if __name__ == "__main__":
    # Example function call
    # create_property_map("your_property_data.xlsx", "./output", "interactive_property_map.html")
    pass
