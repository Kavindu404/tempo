import pandas as pd
import ast
import folium
from shapely import wkt
from shapely.geometry import mapping

def generate_folium_map(excel_path: str, output_html: str = "polygons_map.html"):
    """
    Generates an interactive Folium HTML map showing Ecopia, PRCL, and BLDG polygons
    with separate colors, pins for addresses, and IoU-based tier grouping.
    """
    # Load data
    df = pd.read_excel(excel_path)

    # Convert WKT strings to Shapely geometries
    df["Ecopia_geom"] = df["Ecopia_wkt"].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
    df["PRCL_geom"] = df["PRCL_WKT"].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
    df["BLDG_geom"] = df["BLDG_WKT"].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)

    # Parse lat/lon columns from string lists
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

    # Assign tiers based on IoU
    def assign_tier(iou):
        if pd.isna(iou):
            return "Tier 3"
        if iou > 0.75:
            return "Tier 1"
        elif iou >= 0.5:
            return "Tier 2"
        else:
            return "Tier 3"

    df["Tier"] = df["Ecopia_to_prcl"].apply(assign_tier)

    # Initialize Folium map (centered around median coords if available)
    if df["lat"].notna().any() and df["lon"].notna().any():
        map_center = [df["lat"].median(), df["lon"].median()]
    else:
        map_center = [37.0902, -95.7129]  # default: center of US
    m = folium.Map(location=map_center, zoom_start=12, tiles="cartodbpositron")

    # Colors for polygons
    color_map = {
        "Ecopia": "blue",
        "PRCL": "green",
        "BLDG": "red"
    }

    # Tier color mapping for pins
    tier_color_map = {
        "Tier 1": "darkgreen",
        "Tier 2": "orange",
        "Tier 3": "red"
    }

    # Add polygons to map
    for idx, row in df.iterrows():
        # Plot Ecopia polygon
        if row["Ecopia_geom"] and not row["Ecopia_geom"].is_empty:
            coords = mapping(row["Ecopia_geom"])["coordinates"]
            folium.GeoJson(
                data={
                    "type": "Feature",
                    "geometry": {
                        "type": row["Ecopia_geom"].geom_type,
                        "coordinates": coords,
                    },
                },
                style_function=lambda x, color=color_map["Ecopia"]: {
                    "fillColor": color,
                    "color": color,
                    "weight": 2,
                    "fillOpacity": 0.2,
                },
                name="Ecopia",
                tooltip=f"Ecopia Polygon - Row {idx}"
            ).add_to(m)

        # Plot PRCL polygon
        if row["PRCL_geom"] and not row["PRCL_geom"].is_empty:
            coords = mapping(row["PRCL_geom"])["coordinates"]
            folium.GeoJson(
                data={
                    "type": "Feature",
                    "geometry": {
                        "type": row["PRCL_geom"].geom_type,
                        "coordinates": coords,
                    },
                },
                style_function=lambda x, color=color_map["PRCL"]: {
                    "fillColor": color,
                    "color": color,
                    "weight": 2,
                    "fillOpacity": 0.2,
                },
                name="PRCL",
                tooltip=f"PRCL Polygon - Row {idx}"
            ).add_to(m)

        # Plot BLDG polygon
        if row["BLDG_geom"] and not row["BLDG_geom"].is_empty:
            coords = mapping(row["BLDG_geom"])["coordinates"]
            folium.GeoJson(
                data={
                    "type": "Feature",
                    "geometry": {
                        "type": row["BLDG_geom"].geom_type,
                        "coordinates": coords,
                    },
                },
                style_function=lambda x, color=color_map["BLDG"]: {
                    "fillColor": color,
                    "color": color,
                    "weight": 2,
                    "fillOpacity": 0.2,
                },
                name="BLDG",
                tooltip=f"BLDG Polygon - Row {idx}"
            ).add_to(m)

        # Add pin marker if lat/lon exists
        if pd.notna(row["lat"]) and pd.notna(row["lon"]):
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color=tier_color_map[row["Tier"]],
                fill=True,
                fill_color=tier_color_map[row["Tier"]],
                fill_opacity=0.8,
                tooltip=f"Row {idx} | Tier: {row['Tier']} | IoU: {row['Ecopia_to_prcl']:.2f}"
            ).add_to(m)

    # Add layer control to toggle polygons
    folium.LayerControl().add_to(m)

    # Save map
    m.save(output_html)
    return m
