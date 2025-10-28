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
