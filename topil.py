from psd_tools import PSDImage

def extract_arrow_direction(psd_file, arrow_layer_name):
    # Load the PSD image
    psd = PSDImage.open(psd_file)

    # Find the arrow layer
    arrow_layer = None
    for layer in psd.layers:
        if layer.name == arrow_layer_name:
            arrow_layer = layer
            break

    if arrow_layer:
        # Get the arrow layer as a numpy array
        arrow_image = arrow_layer.topil()

        # Process the arrow layer to extract the direction
        # ...

        # Calculate the angle of the arrow direction
        angle = 0.0  # Placeholder, replace with your code

        print(f"The arrow direction angle is: {angle} degrees")
    else:
        print(f"Arrow layer '{arrow_layer_name}' not found in the PSD file")

# Example usage
psd_file = 'input.psd'  # Input PSD file path
arrow_layer_name = 'Arrow'  # Arrow layer name in the PSD

extract_arrow_direction(psd_file, arrow_layer_name)
