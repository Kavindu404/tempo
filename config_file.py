import os
import json
import math
import random
import csv
from datetime import datetime

# -------------------------
# CONFIG
# -------------------------
# Path to your main directory that contains chunk1, chunk2, ..., chunk130
MAIN_DIR = r"/path/to/main_dir"  # <-- change this

# How many images per category to sample
SAMPLES_PER_CATEGORY = 500

# Date constraint (inclusive)
MIN_DATE = datetime(2021, 1, 1).date()  # "after 2021 Jan" – adjust if you want strictly >


# -------------------------
# TILE -> LAT/LON HELPERS (Web Mercator / Slippy tiles assumption)
# -------------------------
def tile_x_to_lon(x: int, z: int) -> float:
    """Convert tile X to longitude in degrees for zoom z."""
    return x / (2 ** z) * 360.0 - 180.0


def tile_y_to_lat(y: int, z: int) -> float:
    """Convert tile Y to latitude in degrees for zoom z."""
    n = math.pi - 2.0 * math.pi * y / (2 ** z)
    return math.degrees(math.atan(math.sinh(n)))


def tilebox_to_bounds(top_tile: int, left_tile: int, bottom_tile: int, right_tile: int, z: int):
    """
    Convert tilebox indices to [lat_min, lon_min, lat_max, lon_max].

    Assumptions:
    - num1 = top tile index (y_top)
    - num2 = left tile index (x_left)
    - num3 = bottom tile index (y_bottom)
    - num4 = right tile index (x_right)
    - Using standard Web Mercator tile scheme (Slippy tiles).
    """
    # Top edge latitude (north) from y_top
    lat_top = tile_y_to_lat(top_tile, z)
    # Bottom edge latitude (south) from y_bottom + 1 (edge of that tile)
    lat_bottom = tile_y_to_lat(bottom_tile + 1, z)

    # Left edge longitude from x_left
    lon_left = tile_x_to_lon(left_tile, z)
    # Right edge longitude from x_right + 1 (edge of that tile)
    lon_right = tile_x_to_lon(right_tile + 1, z)

    lat_min = min(lat_top, lat_bottom)
    lat_max = max(lat_top, lat_bottom)
    lon_min = min(lon_left, lon_right)
    lon_max = max(lon_left, lon_right)

    return [lat_min, lon_min, lat_max, lon_max]


# -------------------------
# FILENAME PARSER
# -------------------------
def parse_filename(file_name: str):
    """
    Parse file name of the form:
    chunk{xxx}_num1_num2_num3_num4_z{zoom_level}_yyyy-mm-dd.NM.jpg

    Returns:
        {
            "top": int,
            "left": int,
            "bottom": int,
            "right": int,
            "zoom": int,
            "date": datetime.date,
            "nm_date": str,  # original date part (e.g., '2021-05-10.NM')
        }
    or None if parsing fails.
    """
    base = os.path.basename(file_name)
    # Remove extension(s)
    # e.g. "chunk001_123_456_789_101_z21_2021-05-10.NM.jpg"
    if not base.lower().endswith(".jpg"):
        return None

    name_no_ext = base[:-4]  # strip ".jpg"

    parts = name_no_ext.split("_")
    # Expected pattern:
    # 0: 'chunkXXX'
    # 1: num1 (top)
    # 2: num2 (left)
    # 3: num3 (bottom)
    # 4: num4 (right)
    # 5: 'z{zoom}'
    # 6: 'yyyy-mm-dd'
    # 7: 'NM'  (or similar; optional but likely)
    if len(parts) < 7:
        # Not matching expected pattern
        return None

    try:
        # We don't actually need the chunk number, but we could parse it:
        # chunk_id = int(parts[0].replace("chunk", ""))
        top = int(parts[1])
        left = int(parts[2])
        bottom = int(parts[3])
        right = int(parts[4])

        zoom_str = parts[5]
        if not zoom_str.startswith("z"):
            return None
        zoom = int(zoom_str[1:])

        date_str = parts[6]  # 'yyyy-mm-dd'
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()

        # Original NM date: everything after zoom_
        # (e.g. '2021-05-10.NM' or just '2021-05-10' if no NM)
        nm_date = ".".join(parts[6:]) if len(parts) > 6 else date_str

    except Exception:
        return None

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "zoom": zoom,
        "date": date_obj,
        "nm_date": nm_date,
    }


# -------------------------
# MAIN LOGIC
# -------------------------
def collect_sampled_images(main_dir: str, samples_per_category: int = 500):
    random.seed(42)  # for reproducibility

    # category_id -> set of image_names
    category_to_images = {cat: set() for cat in range(10)}

    # image_name -> dict with boundary + nm_date (we'll fill as we parse images)
    image_meta = {}

    # Traverse all chunks and find the JSONs
    for chunk_name in os.listdir(main_dir):
        chunk_path = os.path.join(main_dir, chunk_name)
        if not os.path.isdir(chunk_path):
            continue
        if not chunk_name.lower().startswith("chunk"):
            continue

        json_path = os.path.join(chunk_path, "Predict_Roof_material", "Roof_material_all_in_one.json")
        if not os.path.isfile(json_path):
            print(f"Warning: JSON not found: {json_path}")
            continue

        print(f"Processing {json_path}...")

        with open(json_path, "r") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])

        # Map image_id -> file_name
        image_id_to_fname = {img["id"]: img["file_name"] for img in images}

        # First parse all images in this JSON and store their meta if they pass constraints
        eligible_images = set()
        for img in images:
            file_name = img["file_name"]
            info = parse_filename(file_name)
            if info is None:
                continue

            # Apply constraints: zoom_level > 20 and date >= 2021-01-01
            if info["zoom"] <= 20:
                continue
            if info["date"] < MIN_DATE:
                continue

            # Convert tilebox to bounds
            bounds = tilebox_to_bounds(
                top_tile=info["top"],
                left_tile=info["left"],
                bottom_tile=info["bottom"],
                right_tile=info["right"],
                z=info["zoom"],
            )

            # Store meta keyed by file_name
            image_meta[file_name] = {
                "bounds": bounds,
                "nm_date": info["nm_date"],
            }
            eligible_images.add(file_name)

        # Now go through annotations and add eligible images to category sets
        for ann in annotations:
            cat_id = ann.get("category_id")
            if cat_id not in category_to_images:
                continue  # ignore categories outside 0-9

            img_id = ann.get("image_id")
            file_name = image_id_to_fname.get(img_id)
            if file_name is None:
                continue

            if file_name in eligible_images:
                category_to_images[cat_id].add(file_name)

    # Now sample from each category
    sampled_rows = []
    for cat_id in sorted(category_to_images.keys()):
        images_for_cat = list(category_to_images[cat_id])
        n_available = len(images_for_cat)
        if n_available == 0:
            print(f"Category {cat_id}: 0 eligible images (skipping)")
            continue

        n_sample = min(samples_per_category, n_available)
        print(f"Category {cat_id}: sampling {n_sample} of {n_available} images")
        sampled = random.sample(images_for_cat, n_sample)

        for img_name in sampled:
            meta = image_meta.get(img_name)
            if meta is None:
                # Shouldn't happen, but just in case
                continue
            bounds = meta["bounds"]
            nm_date = meta["nm_date"]
            sampled_rows.append({
                "Image_name": img_name,
                "Category": cat_id,
                "Image_boundary": bounds,  # will be stringified in CSV
                "NM_date": nm_date,
            })

    return sampled_rows


def main():
    sampled_rows = collect_sampled_images(MAIN_DIR, SAMPLES_PER_CATEGORY)

    output_csv = "Sampled Retro Images.csv"
    fieldnames = ["Image_name", "Category", "Image_boundary", "NM_date"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sampled_rows:
            # Convert boundary list to a string
            row_out = row.copy()
            row_out["Image_boundary"] = str(row_out["Image_boundary"])
            writer.writerow(row_out)

    print(f"Saved {len(sampled_rows)} sampled rows to {output_csv}")


if __name__ == "__main__":
    main()









import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Optional, Tuple

# -------------------------------------------------
# Assumed to be defined elsewhere and imported:
# from your_module import ecopia_addr_API
# def ecopia_addr_API(address: str) -> dict:
#     ...
# -------------------------------------------------


def build_address(row: pd.Series) -> str:
    """
    Build an address string from the CSV row:
    "Address1 Address2, City, State ZipCode"
    Skips empty/NaN parts.
    """
    parts = []

    for col in ["Address1", "Address2"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())

    street_part = " ".join(parts)

    city = row.get("City")
    state = row.get("State")
    zipcode = row.get("ZipCode")

    city = str(city).strip() if pd.notna(city) else ""
    state = str(state).strip() if pd.notna(state) else ""
    zipcode = str(zipcode).strip() if pd.notna(zipcode) else ""

    components = []
    if street_part:
        components.append(street_part)
    if city:
        components.append(city)

    state_zip = " ".join([p for p in [state, zipcode] if p])
    if state_zip:
        components.append(state_zip)

    return ", ".join(components)


def fetch_lat_lon_for_row(
    idx: int,
    row: pd.Series,
) -> Tuple[int, Optional[float], Optional[float]]:
    """
    Call ecopia_addr_API for this row and return (index, lat, lon).
    If lookup fails, lat/lon will be None.

    Expects ecopia_addr_API(address) to return a dict like:
    {
        "result": {
            "format_name": "...",
            ...
            "location": {"lat": ..., "lon": ...},
            ...
        },
        "status": True,
        "version": "4.x.x"
    }
    """
    address = build_address(row)

    try:
        resp = ecopia_addr_API(address)  # this should already be resp.json()

        if not isinstance(resp, dict):
            return idx, None, None

        # Check status flag if present
        if resp.get("status") is False:
            return idx, None, None

        result = resp.get("result") or {}
        loc = result.get("location") or {}

        lat = loc.get("lat")
        lon = loc.get("lon")

        if lat is None or lon is None:
            return idx, None, None

        # Sanity checks
        lat = float(lat)
        lon = float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return idx, None, None

        return idx, lat, lon

    except Exception:
        # You can log the exception here if you want
        return idx, None, None


def sample_25_percent(input_csv_path: Path, random_state: int = 42) -> Path:
    """
    Load a CSV, take a 25% random sample, save as *_selected.csv and *_selected.xlsx.
    Returns the path of the *_selected.csv file.
    """
    print(f"\nSampling 25% from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    sampled_df = df.sample(frac=0.25, random_state=random_state)
    print(f"Original rows: {len(df)}, Sampled rows: {len(sampled_df)}")

    selected_csv_path = input_csv_path.with_name(input_csv_path.stem + "_selected.csv")
    selected_xlsx_path = input_csv_path.with_name(input_csv_path.stem + "_selected.xlsx")

    sampled_df.to_csv(selected_csv_path, index=False)
    sampled_df.to_excel(selected_xlsx_path, index=False)

    print(f"Saved sampled CSV to:  {selected_csv_path}")
    print(f"Saved sampled Excel to:{selected_xlsx_path}")

    return selected_csv_path


def geocode_selected_file(selected_csv_path: Path, max_workers: int = 16) -> None:
    """
    Load *_selected.csv, geocode rows missing lat/lon using ecopia_addr_API,
    and save as *_geocoded.csv and *_geocoded.xlsx.
    """
    print(f"\nGeocoding file: {selected_csv_path}")
    df = pd.read_csv(selected_csv_path)

    # Define which rows need geocoding
    # Adjust condition if you only trust hasLatLon or only NaNs
    missing_mask = (
        (df["hasLatLon"] == 0) |
        (df["Latitude"].isna()) |
        (df["Longitude"].isna())
    )

    rows_to_geocode = df[missing_mask]

    print(f"Total rows in selected file: {len(df)}")
    print(f"Rows missing lat/lon to geocode: {len(rows_to_geocode)}")

    if rows_to_geocode.empty:
        print("No rows to geocode. Skipping API calls.")
    else:
        # Multithreaded calls to ecopia_addr_API
        futures = []
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, row in rows_to_geocode.iterrows():
                fut = executor.submit(fetch_lat_lon_for_row, idx, row)
                futures.append(fut)

            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc=f"Geocoding {selected_csv_path.name}"):
                results.append(fut.result())

        # Apply results back to df
        updated_count = 0
        failed_indices = []

        for idx, lat, lon in results:
            if lat is not None and lon is not None:
                df.at[idx, "Latitude"] = lat
                df.at[idx, "Longitude"] = lon
                df.at[idx, "hasLatLon"] = 1
                updated_count += 1
            else:
                failed_indices.append(idx)

        print(f"Successfully updated {updated_count} rows.")
        print(f"Failed to update {len(failed_indices)} rows.")

        # Optional: save failed rows for inspection
        if failed_indices:
            failed_path = selected_csv_path.with_name(
                selected_csv_path.stem + "_geocode_failed.csv"
            )
            df.loc[failed_indices].to_csv(failed_path, index=False)
            print(f"Saved failed rows to: {failed_path}")

    # Save the geocoded DataFrame
    geocoded_csv_path = selected_csv_path.with_name(
        selected_csv_path.stem + "_geocoded.csv"
    )
    geocoded_xlsx_path = selected_csv_path.with_name(
        selected_csv_path.stem + "_geocoded.xlsx"
    )

    df.to_csv(geocoded_csv_path, index=False)
    df.to_excel(geocoded_xlsx_path, index=False)

    print(f"Saved geocoded CSV to:  {geocoded_csv_path}")
    print(f"Saved geocoded Excel to:{geocoded_xlsx_path}")


def main():
    # 🔧 Put your 3 input CSV paths here
    INPUT_FILES = [
        Path("file1.csv"),
        Path("file2.csv"),
        Path("file3.csv"),
    ]

    for csv_path in INPUT_FILES:
        # Step 1: sample 25% and save *_selected
        selected_path = sample_25_percent(csv_path, random_state=42)

        # Step 2: geocode the selected file, save *_geocoded
        geocode_selected_file(selected_path, max_workers=16)


if __name__ == "__main__":
    main()
