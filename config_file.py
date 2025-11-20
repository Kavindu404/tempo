
import os
import pandas as pd
from shapely.geometry import Point, shape
from shapely import wkt

def check_ground_truth(folder1, folder2, excel_file, output_txt):
    """
    folder1: path to folder containing {Full Address}.csv files
    folder2: path to the folder used for false detections
    excel_file: CSV with columns ['Full Address', 'Ground Truth']
    output_txt: where to save missing filenames
    """

    df = pd.read_csv(excel_file)

    # Convert to dictionary for faster lookup
    gt_map = dict(zip(df['Full Address'], df['Ground Truth']))

    missing_list = []

    for fname in os.listdir(folder1):
        fpath = os.path.join(folder1, fname)

        # Skip subfolders
        if not os.path.isfile(fpath):
            continue

        # Expecting filename format "{Full Address}.csv"
        if not fname.lower().endswith(".csv"):
            continue

        full_addr = fname[:-4]  # remove .csv

        # Skip if address not in excel
        if full_addr not in gt_map:
            continue

        ground_truth_raw = str(gt_map[full_addr]).strip()

        # -------------------------------
        # Parse Ground Truth
        # -------------------------------

        if ground_truth_raw == "?":
            # No ground truth → skip
            continue

        # LAT, LON like "(39.35282,-82.52728)"
        if ground_truth_raw.startswith("(") and ground_truth_raw.endswith(")"):
            lat, lon = ground_truth_raw[1:-1].split(",")
            gt_geom = Point(float(lon), float(lat))   # shapely uses (x,y) = (lon,lat)

        # POLYGON or MULTIPOLYGON in WKT
        elif ground_truth_raw.startswith("POLYGON") or ground_truth_raw.startswith("MULTIPOLYGON"):
            gt_geom = wkt.loads(ground_truth_raw)
        else:
            # Unknown format, skip
            continue

        # -------------------------------
        # Load parcel geometry from file
        # -------------------------------

        try:
            df_parcel = pd.read_csv(fpath)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

        if "parcel_geometry" not in df_parcel.columns:
            print(f"Missing parcel_geometry in {fname}")
            continue

        parcel_wkt = df_parcel['parcel_geometry'].iloc[0]

        try:
            parcel_geom = wkt.loads(parcel_wkt)
        except Exception as e:
            print(f"Invalid WKT in {fname}: {e}")
            continue

        # -------------------------------
        # Check containment
        # -------------------------------

        try:
            inside = parcel_geom.contains(gt_geom) or parcel_geom.intersects(gt_geom)
        except Exception:
            inside = False

        if inside:
            # Everything OK
            continue

        # -------------------------------
        # If not inside, check folder2
        # -------------------------------

        alt_path = os.path.join(folder2, fname)

        if os.path.exists(alt_path):
            # Already identified as wrong → skip recording
            continue

        # Otherwise add to missing list
        missing_list.append(fname)

    # -------------------------------
    # Save missing filenames
    # -------------------------------
    with open(output_txt, "w") as f:
        for item in missing_list:
            f.write(item + "\n")

    print(f"Done. Missing count: {len(missing_list)}. Saved to {output_txt}.")




import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

# Example: Create sample data (you'll replace this with your actual image)
def create_sample_contours():
    image = np.zeros((100, 100))
    image[20:40, 20:40] = 1
    image[60:80, 60:80] = 1
    image[30:50, 60:75] = 1
    contours = measure.find_contours(image, level=0.5)
    return image, contours

# Function to plot a specific contour
def plot_contour(contour_idx, image, contours):
    if contours is None or len(contours) == 0:
        return None
    
    # Ensure index is valid
    contour_idx = int(contour_idx)
    if contour_idx >= len(contours):
        contour_idx = len(contours) - 1
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the image
    ax.imshow(image, cmap='gray', alpha=0.5)
    
    # Plot the selected contour
    contour = contours[contour_idx]
    ax.plot(contour[:, 1], contour[:, 0], linewidth=3, color='red')
    
    ax.set_title(f'Contour {contour_idx} (Total: {len(contours)} contours)\n'
                 f'Points in this contour: {len(contour)}')
    ax.axis('image')
    
    plt.tight_layout()
    return fig

# Create the Gradio interface
def create_contour_viewer(image, contours):
    with gr.Blocks() as demo:
        gr.Markdown("# Contour Viewer")
        gr.Markdown(f"Total contours found: **{len(contours)}**")
        
        with gr.Row():
            slider = gr.Slider(
                minimum=0, 
                maximum=len(contours)-1, 
                step=1, 
                value=0, 
                label="Select Contour Index"
            )
        
        plot_output = gr.Plot()
        
        # Update plot when slider changes
        slider.change(
            fn=lambda idx: plot_contour(idx, image, contours),
            inputs=slider,
            outputs=plot_output
        )
        
        # Initial plot
        demo.load(
            fn=lambda: plot_contour(0, image, contours),
            outputs=plot_output
        )
    
    return demo

# Example usage:
image, contours = create_sample_contours()
demo = create_contour_viewer(image, contours)
demo.launch()
