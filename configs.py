import pandas as pd
import ast
import folium
from shapely import wkt
from shapely.geometry import mapping

def plot_rows_by_index(excel_path: str, indexes: list, output_dir: str = "maps_by_index"):
    """
    Generates individual Folium HTML maps for given row indexes,
    showing Ecopia, PRCL, and BLDG polygons with a pin marker.
    
    Parameters:
        excel_path (str): Path to the Excel file with WKT geometries.
        indexes (list): List of row indexes to visualize.
        output_dir (str): Directory where HTML maps will be saved.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load Excel file
    df = pd.read_excel(excel_path)

    # Parse WKT geometries
    df["Ecopia_geom"] = df["Ecopia_wkt"].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
    df["PRCL_geom"] = df["PRCL_WKT"].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
    df["BLDG_geom"] = df["BLDG_WKT"].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)

    # Parse latitude and longitude from string lists
    def safe_parse(coord):
        if pd.isna(coord):
            return None
        try:
            val = ast.literal_eval(coord)
            return float(val[0]) if isinstance(val, list) and val else None
        except:
            return None

    df["lat"] = df["Addr_latitude"].apply(safe_parse)
    df["lon"] = df["Addr_longitude"].apply(safe_parse)

    # Polygon colors
    color_map = {
        "Ecopia": "blue",
        "PRCL": "green",
        "BLDG": "red"
    }

    # Iterate over given indexes
    for idx in indexes:
        if idx not in df.index:
            print(f"⚠️ Index {idx} not found, skipping.")
            continue

        row = df.loc[idx]

        # Center map on lat/lon if available, else use Ecopia polygon centroid, else default
        if pd.notna(row["lat"]) and pd.notna(row["lon"]):
            map_center = [row["lat"], row["lon"]]
        elif row["Ecopia_geom"] and not row["Ecopia_geom"].is_empty:
            centroid = row["Ecopia_geom"].centroid
            map_center = [centroid.y, centroid.x]
        else:
            map_center = [37.0902, -95.7129]  # Default center of the US

        m = folium.Map(location=map_center, zoom_start=18, tiles="cartodbpositron")

        # Helper to add polygon layers
        def add_polygon_layer(geom, label, color):
            if geom and not geom.is_empty:
                folium.GeoJson(
                    data={
                        "type": "Feature",
                        "geometry": {
                            "type": geom.geom_type,
                            "coordinates": mapping(geom)["coordinates"],
                        },
                    },
                    style_function=lambda x, color=color: {
                        "fillColor": color,
                        "color": color,
                        "weight": 2,
                        "fillOpacity": 0.2,
                    },
                    name=label,
                    tooltip=f"{label} Polygon - Row {idx}",
                ).add_to(m)

        # Add Ecopia, PRCL, and BLDG polygons
        add_polygon_layer(row["Ecopia_geom"], "Ecopia", color_map["Ecopia"])
        add_polygon_layer(row["PRCL_geom"], "PRCL", color_map["PRCL"])
        add_polygon_layer(row["BLDG_geom"], "BLDG", color_map["BLDG"])

        # Add pin marker
        if pd.notna(row["lat"]) and pd.notna(row["lon"]):
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color="black",
                fill=True,
                fill_color="yellow",
                fill_opacity=0.9,
                tooltip=f"Row {idx}\nLat: {row['lat']}, Lon: {row['lon']}"
            ).add_to(m)

        # Add layer toggle
        folium.LayerControl().add_to(m)

        # Save map per index
        output_file = os.path.join(output_dir, f"row_{idx}.html")
        m.save(output_file)
        print(f"✅ Saved map for row {idx}: {output_file}")

    print(f"\nAll maps saved in: {output_dir}")
