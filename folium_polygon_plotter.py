import folium
from shapely.wkt import loads
from shapely.geometry import Polygon
import numpy as np

def plot_polygons_folium(wkt_list, map_center=None, zoom_start=10, 
                        polygon_color='blue', polygon_opacity=0.7, 
                        polygon_fill_opacity=0.3, map_width='100%', 
                        map_height='600px'):
    """
    Plot polygons from WKT strings on a Folium map with multiple layer options.
    
    Parameters:
    -----------
    wkt_list : list
        List of WKT strings representing polygons
    map_center : tuple, optional
        (latitude, longitude) for map center. If None, auto-calculates from polygons
    zoom_start : int
        Initial zoom level (default: 10)
    polygon_color : str
        Color of polygon borders (default: 'blue')
    polygon_opacity : float
        Opacity of polygon borders (default: 0.7)
    polygon_fill_opacity : float
        Opacity of polygon fill (default: 0.3)
    map_width : str
        Map width (default: '100%')
    map_height : str
        Map height (default: '600px')
    
    Returns:
    --------
    folium.Map
        Folium map object with polygons and layer controls
    """
    
    # Parse WKT strings to get coordinate bounds for centering
    if map_center is None:
        all_coords = []
        for wkt in wkt_list:
            try:
                geom = loads(wkt)
                if isinstance(geom, Polygon):
                    coords = list(geom.exterior.coords)
                    all_coords.extend(coords)
            except Exception as e:
                print(f"Error parsing WKT: {wkt[:50]}... - {e}")
                continue
        
        if all_coords:
            lons, lats = zip(*all_coords)
            map_center = (np.mean(lats), np.mean(lons))
        else:
            map_center = (39.8283, -98.5795)  # Center of US as fallback
    
    # Create base map with OpenStreetMap
    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        width=map_width,
        height=map_height
    )
    
    # Add different tile layers
    # Google Maps (Normal)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Maps',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Google Satellite
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # OpenStreetMap is already the default base layer
    
    # Add additional useful layers
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Hybrid',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Process and add polygons
    for i, wkt in enumerate(wkt_list):
        try:
            # Parse WKT
            geom = loads(wkt)
            
            if isinstance(geom, Polygon):
                # Convert to lat,lon format for Folium
                coords = [(lat, lon) for lon, lat in geom.exterior.coords]
                
                # Add polygon to map
                folium.Polygon(
                    locations=coords,
                    color=polygon_color,
                    weight=2,
                    opacity=polygon_opacity,
                    fill=True,
                    fillColor=polygon_color,
                    fillOpacity=polygon_fill_opacity,
                    popup=f'Polygon {i+1}',
                    tooltip=f'Polygon {i+1}'
                ).add_to(m)
            else:
                print(f"Warning: Geometry {i+1} is not a polygon: {type(geom)}")
                
        except Exception as e:
            print(f"Error processing polygon {i+1}: {e}")
            continue
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add a marker at the center for reference (optional)
    folium.Marker(
        map_center,
        popup='Map Center',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    return m

# Example usage:
if __name__ == "__main__":
    # Sample WKT polygons
    sample_wkts = [
        'POLYGON((-84.457 38.682, -84.456 38.683, -84.455 38.682, -84.456 38.681, -84.457 38.682))',
        'POLYGON((-84.460 38.685, -84.459 38.686, -84.458 38.685, -84.459 38.684, -84.460 38.685))'
    ]
    
    # Create map
    polygon_map = plot_polygons_folium(
        wkt_list=sample_wkts,
        polygon_color='red',
        polygon_opacity=0.8,
        polygon_fill_opacity=0.4
    )
    
    # Save map
    polygon_map.save('polygon_map.html')
    print("Map saved as 'polygon_map.html'")
