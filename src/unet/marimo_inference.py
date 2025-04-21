

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import marimo as mo
    import numpy as np
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    import torch

    # Import local modules
    import unet_model as unet
    import drawing_bike as pic
    return Image, ImageDraw, mo, np, os, pic, plt, torch, unet


@app.cell
def _(torch):
    # Data Generation Config
    IMG_SIZE = 128 # prev 128 Keep it smaller for faster training initially (e.g., 128x128)
    width, height = IMG_SIZE, IMG_SIZE
    # Training Config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16 # prev 16 Adjust based on your GPU memory

    MODEL_FILE_PATH = "../../models/unet/bike_" # Path to save the trained model
    FILE_VAL_LOSS = f"{MODEL_FILE_PATH}val_loss.pth" # File to save validation loss
    FILE_MODEL = f"{MODEL_FILE_PATH}model.pth" # File to save validation loss

    return BATCH_SIZE, DEVICE, FILE_MODEL, IMG_SIZE, height, width


@app.cell
def _(IMG_SIZE, pic):
    pic.main(IMG_SIZE)
    return


@app.cell
def _(height, mo, width):
    bike_x_slider = mo.ui.slider(0, width, 1, width //2, label="Bike position (x)")
    bike_y_slider = mo.ui.slider(0, height, 1, 100, label="Bike position (y)")
    refresh_background_button = mo.ui.button(value=0, on_click=lambda value: value + 1, label="🏢 Change background")
    refresh_bike_button = mo.ui.button(value=0, on_click=lambda value: value + 1, label="🚲 Change bike")
    noise_slider = mo.ui.slider(0, 1, 0.001, 0, label="Noise")

    return (
        bike_x_slider,
        bike_y_slider,
        noise_slider,
        refresh_background_button,
        refresh_bike_button,
    )


@app.cell
def _(height, np, pic, refresh_background_button, width):
    print(f"refresh background: {refresh_background_button.value}")
    background_img, background_draw, street_y = pic.generate_background(width,height)
    print(f"street_y: {street_y}")
    background_np = np.array(background_img)
    return background_img, street_y


@app.cell
def _(pic, refresh_bike_button, street_y, width):
    print(f"refresh bike: {refresh_bike_button.value}")
    bike = pic.Bike(width, street_y)
    return (bike,)


@app.cell
def _(bike, bike_y_slider):
    bike.set_center_y(bike_y_slider.value)
    return


@app.cell
def _(
    ImageDraw,
    background_img,
    bike,
    bike_x_slider,
    bike_y_slider,
    height,
    noise_slider,
    np,
    pic,
    street_y,
    width,
):
    background_img_copy = background_img.copy()
    background_draw_copy = ImageDraw.Draw(background_img_copy)
    target_x = bike_x_slider.value
    target_y = bike_y_slider.value
    bike_mask_img, bike_mask_draw = pic.add_bike(background_draw_copy, width, height, target_x, target_y, street_y, bike)
    bike_in_town_np_won = np.array(background_img_copy)
    bike_in_town_np = pic.add_noise(bike_in_town_np_won, amount=noise_slider.value)
    bike_mask_np = np.array(bike_mask_img)

    return bike_in_town_np, bike_mask_img


@app.cell
def _(BATCH_SIZE, DEVICE, FILE_MODEL, IMG_SIZE, os, torch, unet):
    # Instantiate the model
    # n_channels=1 (grayscale), n_classes=1 (bike or not bike)
    model = unet.main(device=DEVICE, img_size=IMG_SIZE, n_channels=1, n_classes=1, batch_size=BATCH_SIZE)
    if(os.path.exists(FILE_MODEL)):
        model.load_state_dict(torch.load(FILE_MODEL))
        print(f"Model loaded successfully.")
    else:
        print("No pre-trained model found.")
    return (model,)


@app.cell
def _(DEVICE, Image, bike_in_town_np, model, np, pic, torch):
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        s_input_np = bike_in_town_np
        # 2. Prepare for model (convert to tensor, normalize, add batch dim, move to device)
        s_input_tensor = torch.from_numpy(s_input_np).float().unsqueeze(0).unsqueeze(0) / 255.0
        s_input_tensor = s_input_tensor.to(DEVICE)

        # 3. Get model prediction (logits)
        s_pred_logits = model(s_input_tensor)

        # 4. Convert prediction to probability map (sigmoid) and then to NumPy mask
        s_pred_prob = torch.sigmoid(s_pred_logits)
        s_pred_mask_np = s_pred_prob.squeeze().cpu().numpy() # Remove batch and channel dims

        # 5. Calculate bounding box from the predicted mask
        s_bbox = pic.mask_to_bbox(s_pred_mask_np, threshold=0.5) # Use 0.5 threshold

        # 6. Prepare images for display (convert input back to PIL)
        # Input needs denormalizing and converting back to uint8 if normalized earlier
        # Since we only converted to float and divided by 255, multiply back
        s_display_img_np = (s_input_np).astype(np.uint8)
        s_img_pil = Image.fromarray(s_display_img_np).convert("RGB") # Convert to RGB for red box

        # 7. Draw the bounding box on the PIL image
        s_img_with_bbox = pic.draw_bbox(s_img_pil, s_bbox, color="red", thickness=1)


    return s_img_with_bbox, s_pred_mask_np


@app.cell
def _(
    IMG_SIZE,
    Image,
    bike_in_town_np,
    bike_mask_img,
    bike_x_slider,
    bike_y_slider,
    mo,
    noise_slider,
    np,
    plt,
    refresh_background_button,
    refresh_bike_button,
    s_img_with_bbox,
    s_pred_mask_np,
    street_y,
):
    display_size = IMG_SIZE * 2
    img1 = Image.fromarray(bike_in_town_np).resize((display_size,display_size))
    img2 = bike_mask_img.resize((display_size,display_size))
    img3 = s_img_with_bbox.resize((display_size,display_size))
    mask_color_map = plt.get_cmap("plasma")
    coloring = mask_color_map(s_pred_mask_np)
    img4 = Image.fromarray((coloring[:, :, :3] * 255).astype(np.uint8)).resize((display_size,display_size))
    pictures = mo.hstack([mo.vstack(["Input", img1]),mo.vstack(["TrueMask", img2]),mo.vstack(["Boxed", img3]),mo.vstack(["Predicted mask", img4])])
    positionBlock = mo.vstack([
        mo.hstack([refresh_background_button, refresh_bike_button], justify="start"),
        bike_x_slider, mo.hstack([bike_y_slider,f" ( street is {street_y} )"], justify="start"),
        mo.hstack([noise_slider, "( did learn with 50% at 0.04 )"], justify="start"),
    ])
    display = mo.vstack([mo.md("# Unet at Inference time"), positionBlock, pictures])
    display
    return


if __name__ == "__main__":
    app.run()
