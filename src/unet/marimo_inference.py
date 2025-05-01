

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="full", app_title="Bike Unet Inference")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Inference on a Unet model trained to recognise a bike 🚲""")
    return


@app.cell(hide_code=True)
def import_cell():
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


@app.cell(hide_code=True)
def constant_cell(mo, os, plt, torch):
    # Data Generation Config
    IMG_SIZE = 128 # prev 128 Keep it smaller for faster training initially (e.g., 128x128)
    width, height = IMG_SIZE, IMG_SIZE
    # Training Config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = os.path.join("models","unet")
    # Display Config
    mask_color_map = plt.get_cmap("plasma")
    file_model_prefix_field = mo.ui.text("unet_bike_model", label="Checkpoint file prefix")
    file_model_prefix_field

    return (
        CHECKPOINT_DIR,
        DEVICE,
        IMG_SIZE,
        file_model_prefix_field,
        height,
        mask_color_map,
        width,
    )


@app.cell(hide_code=True)
def file_choice_cell(CHECKPOINT_DIR, file_model_prefix_field, mo, os):
    try:
        _checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(file_model_prefix_field.value) and f.endswith(".pth")]

        if _checkpoint_files:
            _checkpoint_files.sort(
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )  # Sort by epoch number

            _latest_checkpoint = os.path.join(CHECKPOINT_DIR, _checkpoint_files[-1])

    except OSError as e:
        print(f"Could not access checkpoint directory {CHECKPOINT_DIR}: {e}")

    model_file_dropdown = mo.ui.dropdown(
        options=_checkpoint_files, value=_checkpoint_files[-1], label="Choose trained model"
    )
    model_file_dropdown
    return (model_file_dropdown,)


@app.cell(hide_code=True)
def ui_elements_cell(height, mo, width):
    bike_x_slider = mo.ui.slider(0, width, 1, width //2, label="Bike position (x)")
    bike_y_slider = mo.ui.slider(0, height, 1, 100, label="Bike position (y)")
    refresh_background_button = mo.ui.run_button(label="🏢 Change background")
    refresh_bike_button = mo.ui.run_button(label="🚲 Change bike")
    noise_slider = mo.ui.slider(0, 1, 0.001, 0, label="Noise")
    return (
        bike_x_slider,
        bike_y_slider,
        noise_slider,
        refresh_background_button,
        refresh_bike_button,
    )


@app.cell(hide_code=True)
def build_new_background_cell(height, pic, refresh_background_button, width):
    refresh_background_button.value # re-run this cell each time the button is clicked
    background_img, background_draw, street_y = pic.generate_background(width,height)
    return background_img, street_y


@app.cell(hide_code=True)
def build_new_bike_cell(pic, refresh_bike_button, street_y, width):
    refresh_bike_button.value # re-run this cell each time the button is clicked
    bike = pic.Bike(width, street_y)
    return (bike,)


@app.cell(hide_code=True)
def set_bike_y_cell(bike, bike_y_slider):
    # Change bike y position according to the user y choices
    bike.set_center_y(bike_y_slider.value)
    return


@app.cell(hide_code=True)
def build_pic_cell(
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
    # Draw bike and background according to the user choices
    _x = bike_x_slider.value
    _y = bike_y_slider.value
    _noise = noise_slider.value

    _background_img_copy = background_img.copy()
    _background_draw_copy = ImageDraw.Draw(_background_img_copy)
    bike_mask_img, _bike_mask_draw = pic.add_bike(_background_draw_copy, width, height, _x, _y, street_y, bike)
    _bike_in_town_np_won = np.array(_background_img_copy)
    bike_in_town_np = pic.add_noise(_bike_in_town_np_won, amount=_noise)
    return bike_in_town_np, bike_mask_img


@app.cell(hide_code=True)
def build_model_cell(
    CHECKPOINT_DIR,
    DEVICE,
    model_file_dropdown,
    os,
    torch,
    unet,
):
    # Instantiate the model
    # n_channels=1 (grayscale), n_classes=1 (bike or not bike)
    model = unet.UNet(n_channels=1, n_classes=1).to(DEVICE)

    model_file = os.path.join(CHECKPOINT_DIR, model_file_dropdown.value)
    print(f"Model file: {model_file}")
    if(os.path.exists(model_file)):
        _checkpoint = torch.load(model_file, map_location=DEVICE)
        model.load_state_dict(_checkpoint["model_state_dict"])
        print(f"Model loaded successfully.👍")
    else:
        print(f"No pre-trained model found. 🚨 (while looking for {model_file}")
    return (model,)


@app.cell(hide_code=True)
def inference_cell(DEVICE, Image, bike_in_town_np, model, np, pic, torch):
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # 1. Use bike_in_town picture as input
        _input_np = bike_in_town_np

        # 2. Prepare for model (convert to tensor, normalize, add batch dim, move to device)
        _input_tensor = torch.from_numpy(_input_np).float().unsqueeze(0).unsqueeze(0) / 255.0
        _input_tensor = _input_tensor.to(DEVICE)

        # 3. Get model prediction (logits)
        _pred_logits = model(_input_tensor)

        # 4. Convert prediction to probability map (sigmoid) and then to NumPy mask
        _pred_prob = torch.sigmoid(_pred_logits)
        s_pred_mask_np = _pred_prob.squeeze().cpu().numpy() # Remove batch and channel dims

        # 5. Calculate bounding box from the predicted mask
        _bbox = pic.mask_to_bbox(s_pred_mask_np, threshold=0.5) # Use 0.5 threshold

        # 6. Prepare images for display (convert input back to PIL)
        # Input needs denormalizing and converting back to uint8 if normalized earlier
        # Since we only converted to float and divided by 255, multiply back
        _display_img_np = (_input_np).astype(np.uint8)
        _img_pil = Image.fromarray(_display_img_np).convert("RGB") # Convert to RGB for red box

        # 7. Draw the bounding box on the PIL image
        s_img_with_bbox = pic.draw_bbox(_img_pil, _bbox, color="red", thickness=1)

    return s_img_with_bbox, s_pred_mask_np


@app.cell(hide_code=True)
def display_cell(
    IMG_SIZE,
    Image,
    bike_in_town_np,
    bike_mask_img,
    bike_x_slider,
    bike_y_slider,
    mask_color_map,
    mo,
    noise_slider,
    np,
    refresh_background_button,
    refresh_bike_button,
    s_img_with_bbox,
    s_pred_mask_np,
    street_y,
):
    _display_size = IMG_SIZE * 2
    _target_size = (_display_size, _display_size)
    _img1 = Image.fromarray(bike_in_town_np).resize(_target_size)
    _img2 = bike_mask_img.resize(_target_size)
    _img3 = s_img_with_bbox.resize(_target_size)
    _coloring = mask_color_map(s_pred_mask_np)
    _img4 = Image.fromarray((_coloring[:, :, :3] * 255).astype(np.uint8)).resize(_target_size)
    _pictures = mo.hstack([
        mo.vstack(["Input", _img1]),
        mo.vstack(["TrueMask", _img2]),
        mo.vstack(["Boxed", _img3]),
        mo.vstack(["Predicted mask", _img4])])
    _positionBlock = mo.vstack([
        mo.hstack([refresh_background_button, refresh_bike_button], justify="start"),
        bike_x_slider, mo.hstack([bike_y_slider,f" ( street is {street_y} )"], justify="start"),
        mo.hstack([noise_slider, "( did learn with 50% at 0.04 )"], justify="start"),
    ])
    _display = mo.vstack([_positionBlock, _pictures])
    _display
    return


if __name__ == "__main__":
    app.run()
