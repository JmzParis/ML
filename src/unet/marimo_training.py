

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="full", app_title="Bike Unet Training")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # U-Net Training for Simplified Bike Detection with On-the-Fly Data Generation

        ## 🪧Introduction

        This notebook demonstrates how to train a U-Net model to segment simplified drawings of bicycles. The key features are:

        *   **Synthetic Data:** We generate training data programmatically. Each sample consists of:
            *   An input image: A simple urban background sketch (street, buildings, lampposts), a distracting sun sketch, and a simplified bike sketch (2 circles, lines for frame).
            *   A mask image: A binary mask showing *only* the pixels belonging to the bike.
        *   **On-the-Fly Generation:** Data is generated in batches as needed during training, avoiding the need to store a large dataset. This also provides virtually infinite unique training examples.
        *   **Variability:** The bike's position, size, and color (grayscale intensity) vary. The sun's position and brightness also vary. The background details change slightly.
        *   **PyTorch Implementation:** We use PyTorch to define the U-Net architecture and the training loop.
        *   **Goal:** Train the U-Net to accurately segment the bike, ignoring the background clutter and the sun.
        *   **Inference & Visualization:** After training, we demonstrate inference by predicting the mask for a new image and drawing a red bounding box around the detected bike.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🔌Imports

        Import necessary libraries.
        """
    )
    return


@app.cell(hide_code=True)
def import_cell():
    import os

    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    import time

    # Import local modules (cf pyproject.toml for import path)
    import unet_model as unet
    import drawing_bike as pic
    return (
        DataLoader,
        Dataset,
        Image,
        nn,
        np,
        optim,
        os,
        pic,
        plt,
        time,
        torch,
        unet,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🎛️ Configuration

        Define key parameters for data generation and training.
        """
    )
    return


@app.cell(hide_code=True)
def conf_cell(os, torch):
    # Data Generation Config
    IMG_SIZE = 128 # prev 128 Keep it smaller for faster training initially (e.g., 128x128)

    # Training Config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CHECKPOINT_DIR = os.path.join("models","unet") # Path to save the trained model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Model checkpoint folder: {CHECKPOINT_DIR}")
    return CHECKPOINT_DIR, DEVICE, IMG_SIZE


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🖼️Data Generation Functions

        These functions create the synthetic images and masks.
        """
    )
    return


@app.cell(hide_code=True)
def _(generate_learning_picture_button):
    generate_learning_picture_button
    return


@app.cell(hide_code=True)
def demo_pic_cell(IMG_SIZE, generate_learning_picture_button, pic):
    generate_learning_picture_button.value # re-run this cell each time the button is clicked
    pic.main(IMG_SIZE) # Initialize the drawing module
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🎞️ PyTorch Dataset and DataLoader

        We create a custom `Dataset` that uses our `generate_bike_sample` function. The `DataLoader` will then handle batching.
        """
    )
    return


@app.cell(hide_code=True)
def data_cell(
    DataLoader,
    Dataset,
    IMG_SIZE,
    batch_size_slider,
    n_step_slider,
    noise_amount_slider,
    noise_prob_slider,
    os,
    pic,
    torch,
):
    class BikeDataset(Dataset):
        """PyTorch Dataset for generating bike images and masks on-the-fly."""
        def __init__(self, img_size, num_samples):
            self.img_size = img_size
            self.num_samples = num_samples # Effectively, steps per epoch

        def __len__(self):
            # Length is the number of samples we want to generate per epoch
            return self.num_samples

        def __getitem__(self, idx):
            # Generate a new sample each time __getitem__ is called
            input_np, mask_np = pic.generate_bike_sample(self.img_size, self.img_size, noise_prob_slider.value, noise_amount_slider.value)

            # Convert NumPy arrays to PyTorch tensors
            # Input: Add channel dimension (C, H, W) and normalize to [0, 1]
            input_tensor = torch.from_numpy(input_np).float().unsqueeze(0) / 255.0

            # Mask: Add channel dimension and normalize to [0, 1] (for BCE _loss)
            mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0) / 255.0

            return input_tensor, mask_tensor

    # Create DataLoaders for training and validation
    # For validation, we generate a separate set of samples on-the-fly
    _num_workers = 0 if os.name == 'nt' else os.cpu_count()//2  # Zero for Windows compatibility, else use half of available cores
    _dataset = BikeDataset(IMG_SIZE, n_step_slider.value * batch_size_slider.value)
    dataloader = DataLoader(_dataset, batch_size=batch_size_slider.value, shuffle=True, num_workers=_num_workers) # Shuffle batches each epoch

    # Let's check one batch
    try:
        first_batch_img, first_batch_mask = next(iter(dataloader))
        print("Successfully loaded one batch.")
        print(f"Image batch shape: {first_batch_img.shape}") # Should be [BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE]
        print(f"Mask batch shape: {first_batch_mask.shape}")
        print(f"Image batch dtype: {first_batch_img.dtype}") # Should be torch.float32
        print(f"Image batch min/max: {first_batch_img.min():.2f}/{first_batch_img.max():.2f}") # Should be ~0.0/1.0
        print(f"Mask batch min/max: {first_batch_mask.min():.2f}/{first_batch_mask.max():.2f}") # Should be 0.0/1.0
    except Exception as e:
        print(f"Error loading batch: {e}")
        print("Check num_workers or data generation logic if issues persist.")
    return (dataloader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🧠 U-Net Model Definition

        Define the U-Net architecture using standard PyTorch modules.
        """
    )
    return


@app.cell(hide_code=True)
def model_cell(DEVICE, unet):
    # Instantiate the model
    # n_channels=1 (grayscale), n_classes=1 (bike or not bike)
    model = unet.UNet(n_channels=1, n_classes=1).to(DEVICE)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 💾 Training Setup

        Define the _loss function and optimizer. We use `BCEWithLogitsLoss` which is suitable for binary segmentation and expects raw logits from the model.
        """
    )
    return


@app.cell(hide_code=True)
def hyper_param_cell(learning_rate_slider, model, nn, optim):
    # Loss function
    # BCEWithLogitsLoss combines Sigmoid layer and BCELoss in one single class.
    # It's more numerically stable than using a plain Sigmoid followed by BCELoss.
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_slider.value)

    # Learning rate scheduler (optional, but can help)
    # Reduce LR if validation _loss plateaus
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    return criterion, optimizer


@app.cell(hide_code=True)
def ui_elements_cell(mo):
    generate_learning_picture_button = mo.ui.run_button(label="Generate some learning pictures")

    start_training_button = mo.ui.run_button(label="Start/Continue Training")
    n_epochs_slider = mo.ui.slider(1, 100, step=1, value=20, label="Number of Epochs")
    n_step_slider = mo.ui.slider(1, 1000, step=1, value=100, label="Number of Steps per Epochs")
    batch_size_slider = mo.ui.slider(2, 32, step=2, value=16, label="Batch Size")
    learning_rate_slider = mo.ui.number(1e-4, 1e-2, step=1e-5, value=1e-4, label="Learning Rate")

    file_model_prefix_field = mo.ui.text("unet_bike_model", label="Checkpoint file prefix")
    save_freq_epochs_slider = mo.ui.slider(1, 10, value=2, label="Save Checkpoint Every (Epochs)")
    load_checkpoint_flag_checkbox = mo.ui.checkbox(value=True, label="Load latest checkpoint if available")

    noise_prob_slider = mo.ui.slider(0, 1, step=0.1, value=0.5, label="Noise in image Probability")
    noise_amount_slider = mo.ui.slider(0, 1, step=0.01, value=0.04, label="Noise Amount")

    num_inference_samples_slider = mo.ui.slider(0, 20, step=1, value=5, label="Inference samples")
    sep = mo.md('---')
    learning_ui_conf =  mo.vstack([
        mo.hstack([
            mo.vstack([n_epochs_slider, n_step_slider, batch_size_slider, learning_rate_slider]),
            mo.vstack([file_model_prefix_field, save_freq_epochs_slider, load_checkpoint_flag_checkbox], justify="start")
        ]),
        sep,
        noise_prob_slider, noise_amount_slider,
        sep
        ])
    return (
        batch_size_slider,
        file_model_prefix_field,
        generate_learning_picture_button,
        learning_rate_slider,
        learning_ui_conf,
        load_checkpoint_flag_checkbox,
        n_epochs_slider,
        n_step_slider,
        noise_amount_slider,
        noise_prob_slider,
        num_inference_samples_slider,
        save_freq_epochs_slider,
        sep,
        start_training_button,
    )


@app.cell(hide_code=True)
def checkpoint_cell(
    CHECKPOINT_DIR,
    DEVICE,
    file_model_prefix_field,
    load_checkpoint_flag_checkbox,
    mo,
    model,
    optimizer,
    os,
    torch,
):
    def save_model(epoch, avg_epoch_loss):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{file_model_prefix_field.value}_epoch_{epoch}.pth")
        try:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    # Save epoch log for plotting
                    "epoch_log": get_epoch_log(),                 
                },
                checkpoint_path,
            )

            print(f"Checkpoint saved to {checkpoint_path}")

        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    # Use mo.state to manage the training execution and outputs persistently
    # Initialize state variables if they don't exist
    get_epoch_log, set_epoch_log = mo.state([])  # Store tuples of (epoch, avg_loss)
    get_start_epoch, set_start_epoch = mo.state(1)  # Store start_epoch

    _latest_checkpoint = None
    _checkpoint_status = mo.md("No checkpoint found.")
    # Find latest checkpoint file
    try:
        _checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(file_model_prefix_field.value) and f.endswith(".pth")]

        if _checkpoint_files:
            _checkpoint_files.sort(
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )  # Sort by epoch number

            _latest_checkpoint = os.path.join(CHECKPOINT_DIR, _checkpoint_files[-1])

    except OSError as e:
        print(f"Could not access checkpoint directory {CHECKPOINT_DIR}: {e}")

    if load_checkpoint_flag_checkbox.value and _latest_checkpoint:
        try:
            print(f"Loading checkpoint: {_latest_checkpoint}")
            _checkpoint = torch.load(_latest_checkpoint, map_location=DEVICE)
            model.load_state_dict(_checkpoint["model_state_dict"])
            optimizer.load_state_dict(_checkpoint["optimizer_state_dict"])
            set_epoch_log(_checkpoint["epoch_log"])
            _prev_epoch = _checkpoint["epoch"]
            set_start_epoch(_prev_epoch +1)  # Start from next epoch
            # Ensure LR matches current setting if desired, or use saved LR
            # optimizer.param_groups[0]['lr'] = learning_rate_slider.value

            print(f"Checkpoint loaded. Resuming from epoch {_prev_epoch}.")
            _checkpoint_status = mo.md(f"Loaded checkpoint from epoch {_prev_epoch} (Loss: {_checkpoint["loss"]:.4f}).")

        except Exception as e:
            _checkpoint_status = mo.md(f"Error loading checkpoint: {e}").callout(kind="danger")
    elif load_checkpoint_flag_checkbox.value and not _latest_checkpoint:
        _checkpoint_status = mo.md("⚠️ Load requested, but no checkpoint found.")
    else:
        _checkpoint_status = mo.md("Training will start from scratch.")

    _checkpoint_status
    return (
        get_epoch_log,
        get_start_epoch,
        save_model,
        set_epoch_log,
        set_start_epoch,
    )


@app.cell(hide_code=True)
def training_monitor_cell(
    get_epoch_log,
    get_start_epoch,
    learning_ui_conf,
    mo,
    n_epochs_slider,
    num_inference_samples_slider,
    plt,
    sep,
    start_training_button,
):
    # Display training
    def display_epoch_log(epoch_log):
            if not epoch_log:
                return None
            epochs, losses = zip(*epoch_log)
            fig, ax = plt.subplots(figsize=(8, 3))
            plt.ylim(bottom=0,top=losses[0] * 1.2)
            ax.plot(epochs, losses, marker="o", linestyle="-")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss")
            ax.set_title("Training Loss Over Epochs")
            ax.grid(True)
            plt.tight_layout()
            return plt.gcf()

    mo.vstack([
        mo.md('### Training monitor'),
        (mo.vstack([display_epoch_log(get_epoch_log()),f"Last Loss: {get_epoch_log()[-1][1]:.4f}"]) if get_epoch_log() else "No epoch data to display"),
        sep,
        mo.vstack([
            learning_ui_conf, 
            mo.hstack([
                start_training_button,
                (f"From epoch {get_start_epoch()} to {n_epochs_slider.value}" if get_start_epoch() <= n_epochs_slider.value else f"Model is trainned after {get_start_epoch() -1 } epochs. (change epoch number slider to continue: {n_epochs_slider.value}).")
            ]),
        ]),
        num_inference_samples_slider])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤯 Training Loop

        Train the model using the generated data.
        """
    )
    return


@app.cell(hide_code=True)
def training_cell(
    DEVICE,
    criterion,
    dataloader,
    get_epoch_log,
    get_start_epoch,
    mo,
    model,
    n_epochs_slider,
    optimizer,
    save_freq_epochs_slider,
    save_model,
    set_epoch_log,
    set_start_epoch,
    start_training_button,
    time,
):
    # if the button hasn't been clicked, don't run.
    _epoch_to_compute = n_epochs_slider.value + 1 - get_start_epoch()
    mo.stop(not start_training_button.value or _epoch_to_compute <=0, mo.md((f"Press 'Start/Continue Training' button to run {_epoch_to_compute} epochs🔥" if get_start_epoch() <= n_epochs_slider.value else "Model is trained 🟢")))

    print(f"Starting training for {_epoch_to_compute} epochs on {DEVICE}...")

    _steps_per_epoch = len(dataloader)

    _total_start_time = time.time()
    model.train() # Set model to training mode
    for _epoch in range(get_start_epoch(), n_epochs_slider.value + 1):
        _epoch_loss = 0.0
        with mo.status.progress_bar(title=f"Epoch {_epoch}/{n_epochs_slider.value}", total=_steps_per_epoch) as _progress_bar:
            for _step, _batch in enumerate(dataloader):
                # Forward pass
                _images, _masks = _batch
                _outputs = model(_images.to(DEVICE))
                _loss = criterion(_outputs, _masks.to(DEVICE)) # Compare logits with target mask

                # Backward pass and optimize
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                _current_loss = _loss.item()
                _epoch_loss += _current_loss

                # Display progress
                _progress_bar.update(
                    title=f"Epoch {_epoch}/{n_epochs_slider.value} Step {_step + 1}/{_steps_per_epoch}",
                    subtitle=f"Step Batch Loss: {_current_loss:.4f}, Epoch Avg Loss: {_epoch_loss / (_step + 1):.4f}",
                )   

        _avg_epoch_loss = _epoch_loss / _steps_per_epoch
        # Log epoch _loss
        set_epoch_log(get_epoch_log() + [(_epoch, _avg_epoch_loss)])
        print(f"Epoch {_epoch}/{n_epochs_slider.value}, Avg Loss: {_avg_epoch_loss:.4f}")

        set_start_epoch(_epoch+1) # allow restart at next step if this cell is interupted
        # --- Checkpointing ---
        if (_epoch % save_freq_epochs_slider.value == 0 or _epoch == n_epochs_slider.value):
            save_model(_epoch, _avg_epoch_loss)

    # --- Training Finished ---
    _total_end_time = time.time()
    _total_duration = _total_end_time - _total_start_time
    _final_status = f"Training finished after {n_epochs_slider.value} epochs ({_total_duration:.2f}s)."
    print(_final_status)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 📽️Inference and Visualization

        Now, let's use the trained model to predict the mask for a new generated image and draw a bounding box around the detected bike.
        """
    )
    return


@app.cell(hide_code=True)
def inference_cell(
    DEVICE,
    IMG_SIZE,
    Image,
    model,
    np,
    num_inference_samples_slider,
    pic,
    torch,
):
    # Generate a few new samples for inference
    inference_results = []
    num_inference_samples = num_inference_samples_slider.value
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for _ in range(num_inference_samples):
            # 1. Generate new data
            _input_np, _true_mask_np = pic.generate_bike_sample(IMG_SIZE, IMG_SIZE)

            # 2. Prepare for model (convert to tensor, normalize, add batch dim, move to device)
            _input_tensor = torch.from_numpy(_input_np).float().unsqueeze(0).unsqueeze(0) / 255.0
            _input_tensor = _input_tensor.to(DEVICE)

            # 3. Get model prediction (logits)
            _pred_logits = model(_input_tensor)

            # 4. Convert prediction to probability map (sigmoid) and then to NumPy mask
            _pred_prob = torch.sigmoid(_pred_logits)
            _pred_mask_np = _pred_prob.squeeze().cpu().numpy() # Remove batch and channel dims

            # 5. Calculate bounding box from the predicted mask
            _bbox = pic.mask_to_bbox(_pred_mask_np, threshold=0.5) # Use 0.5 threshold

            # 6. Prepare images for display (convert input back to PIL)
            # Input needs denormalizing and converting back to uint8 if normalized earlier
            # Since we only converted to float and divided by 255, multiply back
            _display_img_np = (_input_np).astype(np.uint8)
            _img_pil = Image.fromarray(_display_img_np).convert("RGB") # Convert to RGB for red box

            # 7. Draw the bounding box on the PIL image
            _img_with_bbox = pic.draw_bbox(_img_pil, _bbox, color="red", thickness=1)

            inference_results.append({
                "input_pil": Image.fromarray(_display_img_np), # Original grayscale input
                "true_mask_np": _true_mask_np,
                "pred_mask_np": _pred_mask_np,
                "img_with_bbox": _img_with_bbox,
                "bbox": _bbox
            })
    return inference_results, num_inference_samples


@app.cell(hide_code=True)
def display_cell(inference_results, num_inference_samples, plt):
    # Display the results
    _fig, _axes = plt.subplots(num_inference_samples, 4, figsize=(16, num_inference_samples * 4))
    _fig.suptitle("Inference Results", fontsize=16)

    for _i, _result in enumerate(inference_results):
        _ax_input = _axes[_i, 0]
        _ax_true_mask = _axes[_i, 1]
        _ax_pred_mask = _axes[_i, 2]
        _ax_bbox = _axes[_i, 3]

        _ax_input.imshow(_result["input_pil"], cmap='gray')
        _ax_input.set_title(f"Input Image {_i+1}")
        _ax_input.axis('off')

        _ax_true_mask.imshow(_result["true_mask_np"], cmap='gray')
        _ax_true_mask.set_title("True Mask")
        _ax_true_mask.axis('off')

        _im = _ax_pred_mask.imshow(_result["pred_mask_np"], cmap='viridis', vmin=0, vmax=1) # Show probability map
        _ax_pred_mask.set_title("Predicted Mask (Prob)")
        _ax_pred_mask.axis('off')
        #_fig.colorbar(_im, ax=_ax_pred_mask, fraction=0.046, pad=0.04) # Optional colorbar

        _ax_bbox.imshow(_result["img_with_bbox"])
        _ax_bbox.set_title(f"Prediction w/ BBox\n{_result['bbox']}") # Display coords
        _ax_bbox.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🪦 Conclusion

        This notebook demonstrated the process of training a U-Net for object segmentation using entirely synthetic, on-the-fly generated data.

        **Key takeaways:**

        *   **Synthetic data** can be effective for training, especially for bootstrapping or understanding model behavior on specific features (like shape).
        *   **On-the-fly generation** is memory-efficient and provides a vast amount of training data, reducing overfitting.
        *   **Controlling variability** (object color, distractors) is crucial to ensure the model learns the desired features (shape) rather than simple shortcuts (color intensity).
        *   The U-Net architecture is well-suited for **pixel-wise segmentation tasks**.
        *   The output mask from the U-Net can be easily post-processed (e.g., thresholding, finding contours) to extract higher-level information like **bounding boxes**.

        **Potential improvements:**

        *   More complex backgrounds and bike variations.
        *   Data augmentation (rotation, scaling, elastic deformations) applied to generated samples.
        *   Hyperparameter tuning (learning rate, batch size, network depth/width).
        *   Using more advanced metrics beyond loss (e.g., Dice coefficient, IoU) for evaluation.
        *   Training for longer or using learning rate scheduling.
        """
    )
    return


if __name__ == "__main__":
    app.run()
