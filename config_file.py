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
