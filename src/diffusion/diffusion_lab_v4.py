# -*- coding: utf-8 -*-

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def imports_cell():
    # Keep imports accessible to Marimo's static analysis
    # Re-importing them here within a cell function makes them available to other cells.
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    import torch.nn.functional as TF
    from torchvision.utils import make_grid
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    import math
    import os
    import time
    import random
    from io import BytesIO
    import matplotlib.pyplot as plt
    import traceback
    from typing import Tuple, List, Dict, Optional, Union, Callable

    print("Core libraries imported.")
    return (
        BytesIO,
        Callable,
        DataLoader,
        Dataset,
        Dict,
        Image,
        ImageDraw,
        ImageFilter,
        List,
        Optional,
        T,
        TF,
        Tuple,
        Union,
        make_grid,
        math,
        nn,
        np,
        optim,
        os,
        plt,
        random,
        time,
        torch,
        traceback,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 🖼️ Image Generation using Diffusion Models

        This notebook demonstrates the basics of training and sampling from diffusion models for image generation.
        We will generate synthetic 128x128 images of Platonic solids on a plane on-the-fly.

        **Key Concepts Covered:**  
        - **Forward Process:** Adding noise gradually (q-process).  
        - **Reverse Process:** Learning to remove noise using a U-Net model (p-process).  
        - **Noise Schedules:** Linear vs. Cosine schedules for noise variance.  
        - **Samplers:**  
           - DDPM (Denoising Diffusion Probabilistic Models) and  
           - DDIM (Denoising Diffusion Implicit Models).   
        - **On-the-fly Data Generation:** Creating training data dynamically.  

        **Note:** Training diffusion models is computationally expensive. CPU training will be **very slow**.
        This notebook is primarily for educational purposes. Expect long training times for good results. Use a GPU if available!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🎛️ Configuration""")
    return


@app.cell(hide_code=True)
def ui_elements(mo):
    # --- General Settings ---
    image_size_slider = mo.ui.slider(32, 128, step=32, value=32, label="Image Size (Fixed)")
    batch_size_slider = mo.ui.slider(2, 32, step=2, value=4, label="Batch Size")
    n_epochs_slider = mo.ui.slider(1, 200, step=1, value=10, label="Epochs")
    steps_per_epoch_slider = mo.ui.slider(50, 500, step=10, value=100, label="Steps per Epoch")
    learning_rate_slider = mo.ui.number(1e-5, 1e-2, step=1e-5, value=5e-3, label="Learning Rate")

    # --- Diffusion Settings ---
    timesteps_slider = mo.ui.slider(100, 1000, step=50, value=200, label="Diffusion Timesteps (T)") # Total number of diffusion steps.
    noise_schedule_dropdown = mo.ui.dropdown(["Linear", "Cosine"], value="Linear", label="Noise Schedule")

    # --- Data Generation Settings ---
    color_mode_dropdown = mo.ui.dropdown(["Grayscale", "RGB"], value="Grayscale", label="Image Mode")
    num_solids_range_slider = mo.ui.range_slider(1, 8, value=(1, 2), label="Number of Solids per Image")
    shape_type_dropdown = mo.ui.dropdown(["rectangle", "ellipse", "both"], value="rectangle", label="Shapes")

    # --- Model & Checkpointing ---
    checkpoint_dir_text = mo.ui.text("models\\diffusion", label="Directory")
    load_checkpoint_checkbox = mo.ui.checkbox(value=True, label="Load Latest if Available")
    save_freq_slider = mo.ui.slider(1, 10, value=2, label="Save Every (Epochs)")

    # --- Sampling Controls ---
    sampler_dropdown = mo.ui.dropdown(["DDPM", "DDIM"], value="DDPM", label="Sampler Type")
    num_samples_slider = mo.ui.slider(1, 16, value=4, label="Samples to Generate")

    # --- UI Buttons ---
    generate_data_preview_button = mo.ui.button(label="Preview Generated Data")
    start_training_button = mo.ui.run_button(label="Start / Continue Training")
    generate_samples_button = mo.ui.run_button(label="Generate Samples")

    # --- Layout ---
    ui_config_layout = mo.vstack(
        [
            mo.md("### General & Training"),
            mo.hstack([learning_rate_slider, image_size_slider]),
            mo.hstack(
                [
                    batch_size_slider,
                    n_epochs_slider,
                    steps_per_epoch_slider,
                ]
            ),
            mo.md("### Diffusion Process"),
            mo.hstack([noise_schedule_dropdown, timesteps_slider], justify="start"),
            mo.md("### Data Generation"),
            mo.hstack([color_mode_dropdown, num_solids_range_slider, shape_type_dropdown]),
            mo.md("### Model & Checkpointing"),
            mo.hstack([checkpoint_dir_text, save_freq_slider, load_checkpoint_checkbox]),
            mo.md("---"),
            start_training_button
        ]
    )
    return (
        batch_size_slider,
        checkpoint_dir_text,
        color_mode_dropdown,
        generate_data_preview_button,
        generate_samples_button,
        image_size_slider,
        learning_rate_slider,
        load_checkpoint_checkbox,
        n_epochs_slider,
        noise_schedule_dropdown,
        num_samples_slider,
        num_solids_range_slider,
        sampler_dropdown,
        save_freq_slider,
        shape_type_dropdown,
        start_training_button,
        steps_per_epoch_slider,
        timesteps_slider,
        ui_config_layout,
    )


@app.cell(hide_code=True)
def _(
    batch_size_slider,
    checkpoint_dir_text,
    color_mode_dropdown,
    image_size_slider,
    learning_rate_slider,
    load_checkpoint_checkbox,
    mo,
    n_epochs_slider,
    noise_schedule_dropdown,
    num_solids_range_slider,
    save_freq_slider,
    steps_per_epoch_slider,
    timesteps_slider,
):
    mo.md(
        f"""
        **Current Configuration Summary:**  
         - Image: {image_size_slider.value}x{image_size_slider.value}, Mode: {color_mode_dropdown.value}  
         - Training: {n_epochs_slider.value} epochs, {steps_per_epoch_slider.value} steps/epoch, Batch: {batch_size_slider.value}, LR: {learning_rate_slider.value:.1e}  
         - Diffusion: T={timesteps_slider.value}, Schedule: {noise_schedule_dropdown.value}  
         - Data: {num_solids_range_slider.value[0]}-{num_solids_range_slider.value[1]} solids/image  
         - Checkpointing: Dir: `{checkpoint_dir_text.value}`, Save every {save_freq_slider.value} epochs, Load: {load_checkpoint_checkbox.value}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🧱 Synthetic Data Generation""")
    return


@app.cell(hide_code=True)
def _(Image, ImageDraw, num_solids_range_slider, random, shape_type_dropdown):
    def generate_scene_image(
        img_size: int = 128,
        mode: str = "RGB" # "RGB" or "Grayscale"
    ) -> Image.Image:
        """
        Generates a simple PIL image with basic solid shapes for diffusion model training.

        Args:
            img_size: The width and height of the image in pixels.
            mode: Image mode, "RGB" or "Grayscale".

        Returns:
            A PIL.Image.Image object.
        """
        num_solids_min = num_solids_range_slider.value[0]
        num_solids_max = num_solids_range_slider.value[1]
        min_shape_size_ratio: float = 0.15 # Min size relative to img_size
        max_shape_size_ratio: float = 0.50 # Max size relative to img_size
        fixed_background_color = None
        fixed_palette: list = None # E.g., [(255,0,0), (0,255,0), (0,0,255)] or [0, 128, 255] 
        shape_types: list = None

        if shape_type_dropdown.value == "ellipse":
            shape_types = ["ellipse"]
        elif shape_type_dropdown.value == "rectangle":
            shape_types = ["rectangle"]
        else:
            shape_types = ["rectangle", "ellipse"]

        if mode == "RGB":
            default_bg = (0, 0, 0) # Black background for RGB
            def get_random_color():
                return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            mode = "L" # PIL uses 'L' for grayscale
            default_bg = 0 # Black background for Grayscale
            def get_random_color():
                return random.randint(0, 255)


        background_color = fixed_background_color if fixed_background_color is not None else default_bg
        image = Image.new(mode, (img_size, img_size), background_color)
        draw = ImageDraw.Draw(image)

        num_shapes = random.randint(num_solids_min, num_solids_max)
        min_size_px = int(img_size * min_shape_size_ratio)
        max_size_px = int(img_size * max_shape_size_ratio)

        for _ in range(num_shapes):
            shape_type = random.choice(shape_types)

            # Determine shape color
            if fixed_palette:
                fill_color = random.choice(fixed_palette)
            else:
                fill_color = get_random_color()
                # Ensure contrast with background if possible (simple check)
                if isinstance(fill_color, int) and fill_color == background_color :
                     fill_color = 255 - background_color # Invert for simple contrast
                elif isinstance(fill_color, tuple) and fill_color == background_color:
                     fill_color = (255 - background_color[0], 255 - background_color[1], 255 - background_color[2])


            # Determine shape size and position
            shape_width = random.randint(min_size_px, max_size_px)
            shape_height = shape_width # Keep aspect ratio simple initially, or random.randint(min_size_px, max_size_px)

            # Ensure shape stays mostly within bounds
            # Max top-left corner: img_size - min_dimension (to ensure at least min_dimension is visible)
            # Min top-left corner: -max_dimension + min_dimension (allow partial clipping but ensure some part visible)
            # Simplified: Just place top-left corner within image bounds
            max_x0 = img_size - shape_width
            max_y0 = img_size - shape_height
            x0 = random.randint(0, max_x0 if max_x0 > 0 else 0)
            y0 = random.randint(0, max_y0 if max_y0 > 0 else 0)
            x1 = x0 + shape_width
            y1 = y0 + shape_height

            # Draw the shape
            if shape_type == "rectangle":
                draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=None)
            elif shape_type == "ellipse":
                draw.ellipse([x0, y0, x1, y1], fill=fill_color, outline=None)
            # Add more shapes here if needed (e.g., regular polygons)

        return image
    return (generate_scene_image,)


@app.cell(hide_code=True)
def _(
    BytesIO,
    Dict,
    Image,
    ImageDraw,
    ImageFilter,
    List,
    Tuple,
    Union,
    generate_scene_image,
    math,
    mo,
    np,
    num_solids_range_slider,
    random,
):
    # --- Platonic Solid Definitions (Simplified 2D Projections) ---
    PLATONIC_SOLIDS: Dict[str, List[Tuple[float, float]]] = {
        "Tetrahedron": [(0.0, -0.8), (0.7, 0.4), (-0.7, 0.4)],
        "Cube": [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
        "Octahedron": [(0.0, -0.8), (0.7, 0.0), (0.0, 0.8), (-0.7, 0.0)],
        "Dodecahedron": [
            (0.0, -0.8),
            (0.76, -0.25),
            (0.47, 0.65),
            (-0.47, 0.65),
            (-0.76, -0.25),
        ],
        "Icosahedron": [
            (0.4, -0.7),
            (0.8, 0.0),
            (0.4, 0.7),
            (-0.4, 0.7),
            (-0.8, 0.0),
            (-0.4, -0.7),
        ],
    }
    SOLID_NAMES: List[str] = list(PLATONIC_SOLIDS.keys())

    # --- Colors ---
    VIBRANT_COLORS: Dict[str, Tuple[int, int, int]] = {
        "Tetrahedron": (255, 69, 0),  # OrangeRed
        "Cube": (30, 144, 255),  # DodgerBlue
        "Octahedron": (50, 205, 50),  # LimeGreen
        "Dodecahedron": (255, 215, 0),  # Gold
        "Icosahedron": (153, 50, 204),  # DarkOrchid
    }
    GRAY_COLORS: Dict[str, Tuple[int, int, int]] = {name: (random.randint(100, 200),) * 3 for name in SOLID_NAMES}

    def _get_solid_vertices(solid_name: str, center_x: float, center_y: float, size: float) -> List[Tuple[float, float]]:
        """Scales and translates predefined solid vertices."""
        _vertices = PLATONIC_SOLIDS[solid_name]
        _scaled_vertices = [(center_x + x * size, center_y + y * size) for x, y in _vertices]
        return _scaled_vertices

    def generate_scene_image_old(
        img_size: int = 128,
        mode: str = "RGB"
    ) -> Image.Image:
        """Generates a PIL image with random Platonic solids."""
        num_solids_min = num_solids_range_slider.value[0]
        num_solids_max = num_solids_range_slider.value[1]
        _final_mode = "L" if mode == "Grayscale" else "RGB"
        _num_channels = 1 if _final_mode == "L" else 3

        if _final_mode == "L":
            _background_color: Union[int, Tuple[int, int, int]] = random.randint(30, 70)
        else:
            _background_color = (random.randint(30, 50), random.randint(30, 50), random.randint(50, 70))

        image = Image.new(_final_mode, (img_size, img_size), _background_color)
        draw = ImageDraw.Draw(image)
        _num_solids_to_draw = random.randint(num_solids_min, num_solids_max)
        _placed_solids: List[Tuple[str, float, float, float]] = []

        for _ in range(_num_solids_to_draw):
            _solid_name = random.choice(SOLID_NAMES)
            _max_size = img_size * 0.25
            _min_size = img_size * 0.1
            _solid_size = random.uniform(_min_size, _max_size)

            # Simple placement logic (try a few times)
            _placed = False
            for _attempt in range(10):
                _center_x = random.uniform(_solid_size, img_size - _solid_size)
                _center_y = random.uniform(_solid_size, img_size - _solid_size)
                _too_close = False
                for _, _px, _py, _psize in _placed_solids:
                    _dist = math.sqrt((_center_x - _px) ** 2 + (_center_y - _py) ** 2)
                    if _dist < (_solid_size + _psize) * 0.6:  # Check distance
                        _too_close = True
                        break
                if not _too_close:
                    _placed = True
                    break

            if not _placed:
                continue  # Skip if placement failed

            _vertices = _get_solid_vertices(_solid_name, _center_x, _center_y, _solid_size)

            # --- Colors ---
            if _final_mode == "L":
                _solid_color: Union[int, Tuple[int, int, int]] = GRAY_COLORS[_solid_name][0]
                _outline_color: Union[int, Tuple[int, int, int]] = min(255, _solid_color + 50)
                _shadow_base: int = _background_color
            else:
                _solid_color = VIBRANT_COLORS[_solid_name]
                _outline_color = tuple(min(255, c + 40) for c in _solid_color)
                _shadow_base: Tuple[int, int, int] = _background_color

            # --- Simple Shadow ---
            _shadow_offset_x = _solid_size * 0.1
            _shadow_offset_y = _solid_size * 0.15
            _shadow_vertices = [(x + _shadow_offset_x, y + _shadow_offset_y) for x, y in _vertices]
            _shadow_color = tuple(max(0, c - 20) for c in _shadow_base) if _final_mode == "RGB" else max(0, _shadow_base - 20)

            # Draw shadow (slightly blurred polygon) - Simple approach
            # Create a temporary image for the blurred shadow polygon
            _shadow_layer = Image.new(_final_mode, image.size, 0)  # Transparent if RGBA, black if L/RGB
            _shadow_draw = ImageDraw.Draw(_shadow_layer)
            _shadow_draw.polygon(_shadow_vertices, fill=_shadow_color)
            _shadow_layer = _shadow_layer.filter(ImageFilter.GaussianBlur(radius=_solid_size * 0.08))

            # Alpha composite the shadow layer onto the main image
            # Need to handle L mode carefully if alpha_composite is used (convert to RGBA and back?)
            # Simpler for now: just draw polygon first, then object over it slightly offset
            # draw.polygon(_shadow_vertices, fill=_shadow_color) # <-- simplified alternative

            # Use paste with mask for better blending if possible
            try:
                # Use the blurred shadow intensity as a mask for pasting the shadow color
                _mask = _shadow_layer  # In L mode, this works directly. In RGB, it's trickier.
                if _final_mode == "RGB":
                    # Create a proper alpha mask from the shadow layer for RGB
                    _mask = Image.new("L", image.size, 0)
                    _intensity_threshold = 10  # Only blend where shadow is somewhat visible
                    _shadow_data = np.array(_shadow_layer)
                    _mask_data = np.where(np.mean(_shadow_data, axis=2) > _intensity_threshold, 255, 0).astype(np.uint8)
                    _mask = Image.fromarray(_mask_data)

                _shadow_color_img = Image.new(_final_mode, image.size, _shadow_color)
                image.paste(_shadow_color_img, (0, 0), mask=_mask)
            except Exception:
                # Fallback if masking fails: just draw the polygon
                draw.polygon(_shadow_vertices, fill=_shadow_color)

            # --- Draw Solid ---
            draw.polygon(_vertices, fill=_solid_color, outline=_outline_color, width=1)  # Added outline width
            _placed_solids.append((_solid_name, _center_x, _center_y, _solid_size))

        return image

    def display_sample_images(
        num: int = 4,
        img_size: int = 128,
        mode: str = "RGB"
    ) -> mo.Html:
        """Generates and displays a grid of sample images."""
        images = [generate_scene_image(img_size, mode) for _ in range(num)]
        byte_streams = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            byte_streams.append(buf.getvalue())

        _num_cols = int(math.sqrt(num))
        return mo.hstack([mo.image(bs) for bs in byte_streams], justify="center")

    print("Data generation functions defined.")
    return (display_sample_images,)


@app.cell
def _(generate_data_preview_button):
    generate_data_preview_button
    return


@app.cell(hide_code=True)
def _(
    color_mode_dropdown,
    display_sample_images,
    generate_data_preview_button,
    image_size_slider,
    mo,
):
    mo.stop(generate_data_preview_button.value, mo.md("Click 'Preview Generated Data' to see examples."))

    # Button clicked, generate and display samples
    display_sample_images(
        num=10,
        img_size=image_size_slider.value,
        mode=color_mode_dropdown.value
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🎞️ Dataset and DataLoader""")
    return


@app.cell(hide_code=True)
def build_dataloader(
    Callable,
    DataLoader,
    Dataset,
    Optional,
    T,
    batch_size_slider,
    color_mode_dropdown,
    generate_scene_image,
    image_size_slider,
    mo,
    steps_per_epoch_slider,
    torch,
    traceback,
):
    """Cell for defining on the fly Dataloaders"""

    channels = 3 if color_mode_dropdown.value == "RGB" else 1
    _img_size = image_size_slider.value

    # Define the transformation pipeline
    transform = T.Compose(
        [
            T.ToTensor(),  # PIL Image [0, 255] (H, W, C) or (H,W) -> Torch Tensor [0, 1] (C, H, W)
            T.Lambda(lambda t: (t * 2) - 1),  # Scale [0, 1] -> [-1, 1]
        ]
    )

    class SyntheticSolidsDataset(Dataset):
        """A dataset that generates images on-the-fly."""

        def __init__(
            self,
            img_size: int = 128,
            mode: str = "RGB",
            dataset_size: int = 1000,  # Virtual size
            transform: Optional[Callable] = None
        ):
            self.img_size = img_size
            self.mode = mode
            self.dataset_size = dataset_size
            self.transform = transform
            self.expected_channels = 1 if self.mode == "Grayscale" else 3
            print(f"Dataset initialized: size={dataset_size}, mode={mode}")

        def __len__(self) -> int:
            return self.dataset_size

        def __getitem__(self, idx: int) -> torch.Tensor:
            # Generate image dynamically
            try:
                img = generate_scene_image(self.img_size, self.mode)

                if self.transform:
                    img = self.transform(img)

                # Ensure correct channel dimension for grayscale after ToTensor
                if self.mode == "Grayscale" and img.dim() == 2:
                    img = img.unsqueeze(0)  # (H, W) -> (1, H, W)

                # Sanity check channels
                if img.shape[0] != self.expected_channels:
                    # This should not happen with current logic, but good failsafe
                    print(f"Warning: Image channel mismatch! Index {idx}. Expected {self.expected_channels}, got {img.shape[0]}. Returning zeros.")
                    return torch.zeros(self.expected_channels, self.img_size, self.img_size)

                return img
            except Exception as e:
                print(f"Error generating image at index {idx}: {e}\n{traceback.format_exc()}")
                # Return a dummy tensor on error
                return torch.zeros(self.expected_channels, self.img_size, self.img_size)

    # Calculate virtual dataset size needed for one epoch
    _virtual_dataset_size = steps_per_epoch_slider.value * batch_size_slider.value

    # Instantiate the dataset
    dataset = SyntheticSolidsDataset(
        img_size=_img_size,
        mode=color_mode_dropdown.value,
        dataset_size=_virtual_dataset_size,
        transform=transform
    )

    # Function to create a DataLoader (needed for re-creation if batch size changes)
    def get_dataloader() -> DataLoader:
        _current_batch_size = batch_size_slider.value
        # Re-instantiate dataset with potentially updated config if needed,
        # or assume the existing `dataset` object uses the config correctly.
        # For simplicity, we reuse the `dataset` instance here.
        # Its size is based on the initial config, which is fine for virtual data.
        print(f"Creating DataLoader with batch size={_current_batch_size}...")
        return DataLoader(
            dataset,
            batch_size=_current_batch_size,
            shuffle=True,
            num_workers=0,  # MUST be 0 for Windows + on-the-fly generation
            drop_last=True,  # Important for consistent batch sizes during training
            pin_memory=False,  # Usually False with num_workers=0
        )

    # --- Test DataLoader ---
    _data_loader_status = ""
    _batch_shape = None
    try:
        _temp_loader = get_dataloader()
        _sample_batch = next(iter(_temp_loader))
        _batch_shape = _sample_batch.shape
        _data_loader_status = f"DataLoader ready. Sample batch shape: `{_batch_shape}`"
        del _temp_loader, _sample_batch  # Clean up memory
    except Exception as e:
        _data_loader_status = f"Error creating DataLoader or getting batch: {e}\n{traceback.format_exc()}"
        print(_data_loader_status)  # Also print error to console

    dataloader_info = mo.vstack([mo.md("Dataset and DataLoader configured for on-the-fly generation."), mo.md(f"**DataLoader Status:** {_data_loader_status}")])
    dataloader_info
    return channels, dataloader_info, get_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🧠 Diffusion Model (U-Net)""")
    return


@app.cell(hide_code=True)
def unet_model(
    Optional,
    TF,
    Tuple,
    math,
    mo,
    model_status,
    nn,
    torch,
    traceback,
):
    """Cell to define the U-Net model"""

    # --- Time Embedding ---
    class SinusoidalPosEmb(nn.Module):
        """Sinusoidal Position Embedding Layer"""

        def __init__(self, dim: int):
            super().__init__()
            if dim % 2 != 0:
                raise ValueError("SinusoidalPosEmb dimension must be even.")
            self.dim = dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            half_dim = self.dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
            emb = x[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            return emb

    # --- Building Block: Residual Block with Time Embedding ---
    class ResidualBlock(nn.Module):
        """Residual Block with GroupNorm, SiLU, Conv, Dropout, and optional Time Embedding"""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_emb_dim: Optional[int] = None,
            dropout: float = 0.1,
            groups: int = 8,  # GroupNorm groups
        ):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.norm1 = nn.GroupNorm(groups, out_channels)
            self.act1 = nn.SiLU()  # Swish activation

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.norm2 = nn.GroupNorm(groups, out_channels)
            self.act2 = nn.SiLU()
            self.dropout = nn.Dropout(dropout)

            # Skip connection to match output channels if necessary
            if in_channels != out_channels:
                self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.skip_connection = nn.Identity()

            # MLP for time embedding projection
            if time_emb_dim is not None:
                self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
            else:
                self.time_mlp = None

        def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
            h = self.act1(self.norm1(self.conv1(x)))

            # Add time embedding if provided
            if self.time_mlp is not None and time_emb is not None:
                time_encoding = self.time_mlp(time_emb)
                # Add embedding to feature map channels (broadcast H, W)
                h = h + time_encoding.unsqueeze(-1).unsqueeze(-1)

            h = self.dropout(self.act2(self.norm2(self.conv2(h))))

            # Apply skip connection
            return h + self.skip_connection(x)

    # --- U-Net Architecture ---
    class UNet(nn.Module):
        """U-Net architecture for noise prediction in Diffusion Models"""

        def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            time_emb_dim: int = 128,
            base_dim: int = 32,  # Initial number of channels
            dim_mults: Tuple[int, ...] = (1, 2, 4),  # Channel multipliers per resolution level
            dropout: float = 0.1,
            num_res_blocks: int = 2,  # Number of residual blocks per level
        ):
            super().__init__()
            debug = True

            def printDbg(msg: str) -> None:
                if debug:
                    print(msg)

            printDbg(f"UNet Init: in_channels: {in_channels}, out_channels: {out_channels}, base_dim: {base_dim}, dim_mults: {dim_mults}")
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.base_dim = base_dim
            self.dim_mults = dim_mults

            current_dim = base_dim
            num_resolutions = len(dim_mults)

            # 1. Time Embedding Projection
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.SiLU(),
                nn.Linear(time_emb_dim * 4, time_emb_dim),
            )
            printDbg(f"UNet Init: Time MLP output dim: {time_emb_dim} s:{self.time_mlp.shape}")

            # 2. Initial Convolution
            self.init_conv = nn.Conv2d(in_channels, current_dim, kernel_size=3, padding=1)
            printDbg(f"UNet Init: Initial Conv: {in_channels} -> {current_dim} s:{self.init_conv.shape}")

            # 3. Encoder Path (Downsampling)
            self.downs = nn.ModuleList([])
            encoder_dims = [current_dim]  # Store output dims of each encoder level

            indent = ""
            printDbg("--- UNet Init: Encoder ---")
            for i, mult in enumerate(dim_mults):
                indent = "  " * i
                is_last_level = i == num_resolutions - 1
                dim_out = base_dim * mult

                printDbg(f" {indent} Encoder Level {i}: {current_dim} -> {dim_out} channels")
                blocks = []

                printDbg(f"   {indent} ResBlock: {current_dim} -> {dim_out}")
                blocks.append(ResidualBlock(current_dim, dim_out, time_emb_dim, dropout))

                current_dim = dim_out
                for _ in range(num_res_blocks - 1):
                    printDbg(f"   {indent} ResBlock: {current_dim} -> {current_dim}")
                    blocks.append(ResidualBlock(current_dim, current_dim, time_emb_dim, dropout))

                # Downsampling layer (except for the last encoder level)
                if not is_last_level:
                    printDbg(f"   {indent} Downsample Conv: {current_dim} -> {current_dim}")
                    blocks.append(nn.Conv2d(current_dim, current_dim, kernel_size=3, stride=2, padding=1))
                else:
                    printDbg(f"   {indent} Identity: {current_dim} (as last level)")
                    blocks.append(nn.Identity())  # No downsampling at last level


                self.downs.append(nn.ModuleList(blocks))
                encoder_dims.append(current_dim)


            # 4. Bottleneck
            printDbg(f"--- UNet Init: Bottleneck ({current_dim} channels) ---")
            printDbg(f"   {indent} ResBlock: {current_dim} -> {current_dim}")
            self.mid_block1 = ResidualBlock(current_dim, current_dim, time_emb_dim, dropout)
            printDbg(f"   {indent} ResBlock: {current_dim} -> {current_dim}")
            self.mid_block2 = ResidualBlock(current_dim, current_dim, time_emb_dim, dropout)

            # 5. Decoder Path (Upsampling)
            self.ups = nn.ModuleList([])
            printDbg("--- UNet Init: Decoder ---")
            # Iterate through encoder levels in reverse
            for i, mult in reversed(list(enumerate(dim_mults))): # e.g., i = 2, then i = 1, then i = 0 for mults=(1,2,4)
                indent = "  " * i
                is_first_level = i == 0  # Check if it's the highest resolution level
                is_bottleneck_connection = i == num_resolutions - 1  # Check if connecting to bottleneck
                dim_in = current_dim  # Dimension from previous decoder level or bottleneck
                dim_skip = encoder_dims[i + 1]  # Dimension of skip connection from corresponding encoder level i
                dim_out = base_dim * dim_mults[i - 1] if not is_first_level else base_dim  # Target output dim for this level
                printDbg(f" {indent} Decoder Level {i}: Input Dim (from prev): {dim_in}, Skip Dim: {dim_skip}, Target Out Dim: {dim_out}")

                # Upsampling layer: Upsamples spatially AND adjusts channels from dim_in to dim_out
                # This simplifies the ResBlock input calculation later.
                upsample_layer = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)
                printDbg(f"   {indent} Upsample ConvT: {dim_in} -> {dim_out}")

                # Input dimension to the first ResBlock *after* concatenation will be:
                # channels from Upsample (dim_out) + channels from Skip (dim_skip)
                block_input_dim = dim_out + dim_skip

                level_modules = []
                # Add the Upsample layer FIRST for easier forward pass handling
                level_modules.append(upsample_layer)

                # Add Residual Blocks for this level
                # First ResBlock takes the concatenated dimension, outputs target level dimension
                printDbg(f"   {indent} ResBlock 1: {dim_out} + {dim_skip} (skip) = {block_input_dim} -> {dim_out}")
                level_modules.append(ResidualBlock(block_input_dim, dim_out, time_emb_dim, dropout))

                # Subsequent ResBlocks operate purely on the target level dimension
                for blk_idx in range(num_res_blocks - 1):
                    printDbg(f"   {indent} ResBlock {blk_idx + 2}: {dim_out} -> {dim_out}")
                    level_modules.append(ResidualBlock(dim_out, dim_out, time_emb_dim, dropout))

                self.ups.append(nn.ModuleList(level_modules))
                current_dim = dim_out  # Update current_dim for the next (higher res) level input

            # 6. Final Convolution Layer
            self.final_conv = nn.Sequential(
                nn.GroupNorm(8, base_dim),  # Norm before final conv
                nn.SiLU(),
                nn.Conv2d(base_dim, out_channels, kernel_size=1),
            )
            printDbg(f"--- UNet Init: Final Conv ({base_dim} -> {out_channels})  kernel_size=1---")
            printDbg("--- UNet Init: Complete ---")

        def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            # 1. Get time embedding
            t_emb = self.time_mlp(time)

            # 2. Initial convolution
            h = self.init_conv(x)  # (B, base_dim, H, W)
            skips = [h]  # Store for skip connections (after init_conv)

            # 3. Encoder path
            for i, level_blocks in enumerate(self.downs):
                for block in level_blocks[:-1]:  # Apply ResBlocks
                    h = block(h, t_emb)
                skips.append(h)  # Store output *before* downsampling for skip connection
                downsample_layer = level_blocks[-1]  # Get the downsample/identity layer
                h = downsample_layer(h)

            # 4. Bottleneck
            h = self.mid_block1(h, t_emb)
            h = self.mid_block2(h, t_emb)

            # 5. Decoder path
            # print("--- Decoder Forward ---")
            # h starts with bottleneck output here
            for i, level_module_list in enumerate(self.ups):
                # print(f" Decoder Level {i} Input: {h.shape}")
                upsample_layer = level_module_list[0]  # Upsample layer is first
                res_blocks = level_module_list[1:]  # ResBlocks follow

                h = upsample_layer(h)  # Apply Upsample (changes spatial size and channels)
                # print(f"  After Upsample {upsample_layer.__class__.__name__}: {h.shape}")

                skip_connection = skips.pop()
                # print(f"  Skip Connection popped: {skip_connection.shape}")

                # Pad h if spatial dimensions don't match skip_connection after upsampling
                # (Should be less common if strides/padding are standard)
                diffY = skip_connection.size()[2] - h.size()[2]
                diffX = skip_connection.size()[3] - h.size()[3]
                if diffX != 0 or diffY != 0:
                    # print(f"  Padding needed: X={diffX}, Y={diffY}")
                    h = TF.pad(h, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
                    # print(f"  Padded Upsampled h: {h.shape}")

                # Concatenate upsampled tensor and skip connection
                # Input channels = dim_out (from upsample) + dim_skip
                h = torch.cat((h, skip_connection), dim=1)
                # print(f"  Concatenated h: {h.shape}")  # Check channel dim here

                # Apply Residual Blocks for this level
                for block_idx, block in enumerate(res_blocks):
                    # print(f"    Applying ResBlock {block_idx + 1} ({block.conv1.in_channels} -> {block.conv1.out_channels})")
                    h = block(h, t_emb)
                    # print(f"      Output h: {h.shape}")

            # 6. Final convolution
            # print(f"--- Final Conv Input ({h.shape}) ---")
            out = self.final_conv(h)
            # print(f"--- Final Output ({out.shape}) ---")
            return out

    def test_intanciating(base_dim: int, dim_mults: Tuple[int, ...]) -> str:
        # === Code to Instantiate and Test (Lightweight) ===
        status = "Model definition ready."
        try:
            # Simple test case
            dim_mults=(1, 2) # (1, 2, 4)
            base_dim = 32 # 16
            test_unet = UNet(in_channels=3, out_channels=3, base_dim=base_dim, dim_mults=dim_mults)
            test_params = sum(p.numel() for p in test_unet.parameters())
            status = f"U-Net class defined. Test instance (base_dim={base_dim}, mults={dim_mults}) has {test_params:,} parameters."
            del test_unet  # clean up
        except Exception as e:
            status = f"Error during U-Net class definition or test: {e}"
            print(f"\nERROR: {model_status}\n{traceback.format_exc()}")
            return status


    _base_dim = 16 # 32
    _dim_mults=(1, 2) # (1, 2, 4)
    _model_status = test_intanciating(_base_dim, _dim_mults)
    _model_info = mo.vstack([mo.md("U-Net model components (`SinusoidalPosEmb`, `ResidualBlock`, `UNet`) defined."), mo.md(f"**Status:** {_model_status}")])
    _model_info
    return (UNet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🍵🪜 Diffusion steps
        Base on noise schedule pre compute $\alpha_t$, $\beta_t$, $\bar{\alpha}_t = \prod_{i=1}^t (1-\beta_i)$.  
        These Noise Schedules will be later used for Sampling steps
        """
    )
    return


@app.cell(hide_code=True)
def diffusion_tools(
    Optional,
    Tuple,
    mo,
    noise_schedule_dropdown,
    timesteps_slider,
    torch,
):
    """Cell to store diffusion utilities methods for DDPM and DDIM"""

    # --- Noise Schedule Functions ---
    def linear_beta_schedule(timesteps: int) -> torch.Tensor:
        """
        Standard linear schedule from DDPM paper.

        Args:
            timesteps (int): Total number of diffusion steps.

        Returns:
            torch.Tensor: Tensor of beta values [T].
        """
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule from 'Improved Denoising Diffusion Probabilistic Models'.

        Args:
            timesteps (int): Total number of diffusion steps.
            s (float): Offset parameter, default is 0.008.

        Returns:
            torch.Tensor: Tensor of beta values [T].
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # Calculate betas based on adjacent alpha_cumprod values
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Clip betas to prevent numerical instability (e.g., division by zero)
        return torch.clip(betas, 0.0001, 0.9999)

    # --- Select Schedule Based on UI ---
    timesteps = timesteps_slider.value
    _schedule_name = noise_schedule_dropdown.value

    if _schedule_name == "Linear":
        betas = linear_beta_schedule(timesteps)
    elif _schedule_name == "Cosine":
        betas = cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown noise schedule: {_schedule_name}")

    # Ensure calculations are done on CPU initially, can be moved to device later if needed
    betas = betas.cpu() # Make sure calculations happen on CPU unless GPU is explicitly needed

    # --- Pre-calculate Diffusion Constants ---
    alphas: torch.Tensor = 1.0 - betas
    alphas_cumprod: torch.Tensor = torch.cumprod(alphas, axis=0)
    # Pad with 1.0 at the beginning for calculation at t=0, for calculation convenience (alpha_cumprod_0 = 1)
    alphas_cumprod_prev: torch.Tensor = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]])

    # Required for q_sample (forward process)
    sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(1.0 - alphas_cumprod)

    # Required for DDPM p_sample (reverse process)
    sqrt_recip_alphas: torch.Tensor = torch.sqrt(1.0 / alphas)
    # This is beta_tilde_t (variance of q(x_{t-1} | x_t, x_0))
    posterior_variance: torch.Tensor = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)  # beta_tilde_t
    # Ensure no NaN from potential division by zero at t=0 (1-alphas_cumprod[0] can be 0)
    posterior_variance[0] = betas[0] * (1.0 - 1.0) / (1.0 - alphas_cumprod[0]) # Should still be 0 if alphas_cumprod[0] is not exactly 1, handle carefully if needed
    # Or more robustly:
    # Clamp denominator to avoid division by zero or very small numbers if necessary
    # posterior_variance = betas * (1.0 - alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-6)

    # --- Move constants to default device if using GPU ---
    # Example: default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # betas = betas.to(default_device)
    # alphas = alphas.to(default_device)
    # ... etc for all constants ...

    print(f"Diffusion constants calculated for T={timesteps} using {_schedule_name} schedule.")
    print(f"  betas shape: {betas.shape}, range: ({betas.min():.4f}, {betas.max():.4f})")
    print(f"  alphas_cumprod shape: {alphas_cumprod.shape}, range: ({alphas_cumprod.min():.4f}, {alphas_cumprod.max():.4f})")

    # --- Helper Function to Extract Values for a Batch ---
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Extracts values from schedule tensor 'a' at indices 't' and reshapes
        for broadcasting to the shape of 'x'.

        Args:
            a (torch.Tensor): The schedule tensor (e.g., betas, alphas_cumprod) [T].
            t (torch.Tensor): Tensor of timesteps for the batch [batch_size].
            x_shape (Tuple[int, ...]): The shape of the target tensor 'x' (e.g., [B, C, H, W]).

        Returns:
            torch.Tensor: Tensor with extracted values, reshaped for broadcasting.
                          Shape: [batch_size, 1, 1, ...] matching x_shape rank.
        """
        batch_size = t.shape[0]
        # Use t's device for indexing, ensure 'a' is on the same device or CPU
        # Gather using t.long() as indices
        # Ensure 'a' is on the same device as 't' before gathering
        out = a.to(device=t.device).gather(0, t.long())
        # Reshape to (batch_size, 1, 1, 1) to match rank of x_shape for broadcasting with image shape
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # --- Forward Process (q - adding noise) ---
    # q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
    def q_sample(x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Samples x_t by adding noise to x_0 according to the diffusion schedule.
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)

        Args:
            x_start (torch.Tensor): The initial data (e.g., image) [B, C, H, W].
            t (torch.Tensor): Timesteps for the batch [B].
            noise (Optional[torch.Tensor]): Optional pre-generated noise. If None,
                                             Gaussian noise is sampled. [B, C, H, W].

        Returns:
            torch.Tensor: The noised sample x_t [B, C, H, W].
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # Apply the forward process formula
        # equation: sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # --- Display Info ---
    schedule_info = mo.md(f"""
    Diffusion constants calculated for **T = {timesteps}** steps using the **{_schedule_name}** schedule.  
     - `betas`: Shape `{betas.shape}`, Range `({betas.min():.4f}, {betas.max():.4f})`  
     - `alphas_cumprod`: Shape `{alphas_cumprod.shape}`, Range `({alphas_cumprod.min():.4f}, {alphas_cumprod.max():.4f})`  
    """)
    schedule_info
    return (
        alphas_cumprod,
        alphas_cumprod_prev,
        betas,
        extract,
        q_sample,
        sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas,
        timesteps,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 💾 Training Setup (Model Instantiation, Optimizer, Checkpointing)""")
    return


@app.cell(hide_code=True)
def _(
    color_mode_dropdown,
    image_size_slider,
    mo,
    noise_schedule_dropdown,
    timesteps_slider,
):
    checkpoint_file_prefix_text = mo.ui.text(f"chk_s{image_size_slider.value}_{color_mode_dropdown.value}_{timesteps_slider.value}t_{noise_schedule_dropdown.value}", label="Checkpoint file prefix")
    checkpoint_file_prefix_text
    return (checkpoint_file_prefix_text,)


@app.cell(hide_code=True)
def checkpoints(
    Dict,
    Optional,
    checkpoint_dir_text,
    checkpoint_file_prefix_text,
    load_checkpoint_checkbox,
    mo,
    nn,
    noise_schedule_dropdown,
    optim,
    os,
    plt,
    timesteps_slider,
    torch,
    traceback,
):
    """Cell for managing model checkpoints: saving, loading, and plotting loss."""

    _chkpt_dir = checkpoint_dir_text.value
    _file_prefix = checkpoint_file_prefix_text.value
    _load_flag = load_checkpoint_checkbox.value

    # --- Marimo State for Persistent Training Info ---
    # Stores list of (epoch, avg_loss) tuples
    get_epoch_log, set_epoch_log = mo.state([])
    # Stores the epoch number to start training from (1 if new, N+1 if loaded)
    get_start_epoch, set_start_epoch = mo.state(1)

    def display_loss_plot() -> Optional[plt.Figure]:
        """Renders a plot of the training loss stored in Marimo state."""
        epoch_log = get_epoch_log()
        if not epoch_log:
            return None  # Don't display plot if no log exists
        try:
            epochs, losses = zip(*epoch_log)
            fig, ax = plt.subplots(figsize=(8, 3))
            # Dynamically adjust y-axis limit based on initial loss
            y_top_limit = losses[0] * 1.2 if losses else 1.0
            plt.ylim(bottom=0, top=max(0.1, y_top_limit))  # Ensure non-negative limit
            ax.plot(epochs, losses, marker="o", linestyle="-", markersize=4, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss")
            ax.set_title("Training Loss Over Epochs")
            ax.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            # plt.show() # Not needed in Marimo, just return the figure
            return fig  # Return the matplotlib figure object
        except Exception as e:
            print(f"Error plotting loss: {e}")
            return mo.md(f"Error plotting loss: {e}").callout(kind="danger")

    def save_checkpoint(
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        avg_epoch_loss: float,
        config: Dict,  # Store relevant config settings
    ) -> None:
        """Saves model, optimizer state, epoch, loss, and config to a file."""
        os.makedirs(_chkpt_dir, exist_ok=True)  # Ensure directory exists
        # Use a consistent filename structure
        checkpoint_path = os.path.join(_chkpt_dir, f"{_file_prefix}_epoch_{epoch}.pth")
        try:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "epoch_log": get_epoch_log(),  # Save the loss history
                    "config": config,  # Save relevant configuration
                },
                checkpoint_path,
            )
            print(f"✅ Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")

    def _read_checkpoint_data(checkpoint: Dict, model: nn.Module, optimizer: optim.Optimizer) -> str:
        """Helper to load state dicts and update Marimo state."""
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        prev_log = checkpoint.get("epoch_log", [])  # Handle older checkpoints
        set_epoch_log(prev_log)
        prev_epoch = checkpoint["epoch"]
        set_start_epoch(prev_epoch + 1)  # Resume from the next epoch

        # Option: uncomment to force LR to current UI setting after loading
        # current_lr = learning_rate_slider.value
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = current_lr

        config_info = checkpoint.get("config", {})
        msg = (
            f"👍 Checkpoint loaded from epoch {prev_epoch} "
            f"(Loss: {checkpoint['loss']:.4f}). "
            f"Config: T={config_info.get('T', 'N/A')}, "
            f"Schedule={config_info.get('schedule', 'N/A')}. "
            f"Resuming from epoch {get_start_epoch()}."
        )
        print(msg)
        return msg

    def load_latest_checkpoint(model: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> str:
        """Finds and loads the latest valid checkpoint file from the directory."""
        status_msg = "▶️ Starting training from scratch."
        set_start_epoch(1)  # Reset start epoch
        set_epoch_log([])  # Reset epoch log

        if not _load_flag:
            return "Load checkpoint disabled. Starting from scratch."

        if not os.path.isdir(_chkpt_dir):
            return f"⚠️ Checkpoint directory not found: `{_chkpt_dir}`. Starting from scratch."

        try:
            # Find files matching a pattern (e.g., *epoch_*.pth)
            files = [f for f in os.listdir(_chkpt_dir) if f.startswith(_file_prefix) and f.endswith(".pth") and "_epoch_" in f]

            if not files:
                return f"ℹ️ In '{_chkpt_dir}' no '{_file_prefix}_epoch_N.pth' file found.<br>No checkpoint found => Starting from scratch."

            # Sort files by epoch number (assuming format like ..._epoch_N.pth)
            files.sort(key=lambda x: int(x.split("_epoch_")[-1].split(".")[0]), reverse=True)
            latest_file_path = os.path.join(_chkpt_dir, files[0])

            print(f"Attempting to load checkpoint: {latest_file_path}")
            checkpoint = torch.load(latest_file_path, map_location=device)

            # --- Configuration Compatibility Check (Optional but Recommended) ---
            # Compare crucial loaded config aspects with current UI settings
            loaded_config = checkpoint.get("config", {})
            mismatches = []
            # Example checks (add more as needed):
            if loaded_config.get("T") != timesteps_slider.value:
                mismatches.append(f"Timesteps (T): Loaded={loaded_config.get('T')} vs UI={timesteps_slider.value}")
            if loaded_config.get("schedule") != noise_schedule_dropdown.value:
                mismatches.append(f"Schedule: Loaded={loaded_config.get('schedule')} vs UI={noise_schedule_dropdown.value}")
            if mismatches:
                warning = "⚠️ Checkpoint loaded, but config mismatches detected:\n - " + "\n - ".join(mismatches) + "\n Training will continue, but results might be unexpected."
                print(warning)
                status_msg = _read_checkpoint_data(checkpoint, model, optimizer) + "\n" + warning
            else:
                status_msg = _read_checkpoint_data(checkpoint, model, optimizer)

        except FileNotFoundError:
            status_msg = "⚠️ Checkpoint file specified but not found. Starting from scratch."
        except Exception as e:
            status_msg = f"❌ Error loading checkpoint from `{files[0]}`: {e}. Training from scratch."
            set_start_epoch(1)  # Reset state if loading failed
            set_epoch_log([])
            print(f"{status_msg}\n{traceback.format_exc()}")

        return status_msg

    return (
        display_loss_plot,
        get_epoch_log,
        get_start_epoch,
        load_latest_checkpoint,
        save_checkpoint,
        set_epoch_log,
        set_start_epoch,
    )


@app.cell(hide_code=True)
def training_setup(
    UNet,
    channels,
    dataloader_info,
    display_loss_plot,
    learning_rate_slider,
    load_latest_checkpoint,
    mo,
    nn,
    optim,
    torch,
):
    """Cell to define training setup: Instanciating model, optimizer, loss_fn, reading prev checkpoints"""
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU. Training will be slow.")

    # --- Instantiate Model ---
    # Use configuration values passed from previous cells
    _base_dim = 16 # 32
    _dim_mults = (1, 2) # (1, 2, 4)
    _num_res_blocks = 2

    model = UNet(
        in_channels=channels,
        out_channels=channels,
        time_emb_dim=128,  # Can be adjusted, ensure consistency if loading checkpoints
        base_dim=_base_dim,  # Start with 32, maybe increase to 64 if using GPU and need more capacity
        dim_mults=_dim_mults,  # (1, 2) for faster/lighter, (1, 2, 4) for more capacity
        num_res_blocks=_num_res_blocks,
        dropout=0.1,
    ).to(device)

    _model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated with (base_dim={_base_dim}, mults={_dim_mults}, res_blocks{_num_res_blocks}) => {_model_param_count:,} trainable parameters.")

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate_slider.value, weight_decay=1e-4)
    print(f"Optimizer: AdamW, LR={learning_rate_slider.value:.1e}")

    # --- Loss Function ---
    loss_fn = nn.MSELoss()  # Standard loss for predicting noise
    print("Loss Function: MSELoss")

    # --- Load Checkpoint (if enabled and available) ---
    # Pass the current device to map the loaded tensors correctly
    _checkpoint_status_msg = load_latest_checkpoint(model, optimizer, device)

    # Display setup status in Marimo UI
    training_setup_info = mo.vstack(
        [
            dataloader_info,  # Show DataLoader status first
            mo.md("---"),
            mo.md(f"**Device:** `{device}`"),
            mo.md(f"**Model:** U-Net with {_model_param_count:,} parameters. (base_dim={_base_dim}, mults={_dim_mults}, res_blocks{_num_res_blocks})"),
            mo.md(f"**Optimizer:** AdamW (LR={learning_rate_slider.value:.1e})"),
            mo.md("**Loss Function:** MSELoss"),
            mo.md(f"**Checkpoint Status:** {_checkpoint_status_msg}"),
            mo.md("---"),
            mo.md("**Training Loss History:**"),
            display_loss_plot(),  # Display the loss plot
        ]
    )
    training_setup_info
    return device, loss_fn, model, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🚀 Training Loop""")
    return


@app.cell(hide_code=True)
def display_ui_conf(ui_config_layout):
    # Display the main config UI again for easy access
    ui_config_layout
    return


@app.cell(hide_code=True)
def training_loop(
    DataLoader,
    device,
    get_dataloader,
    get_epoch_log,
    get_start_epoch,
    loss_fn,
    mo,
    model,
    n_epochs_slider,
    noise_schedule_dropdown,
    optimizer,
    q_sample,
    save_checkpoint,
    save_freq_slider,
    set_epoch_log,
    set_start_epoch,
    start_training_button,
    steps_per_epoch_slider,
    time,
    timesteps,
    torch,
    traceback,
):
    # Only run this cell if the "Start Training" button was clicked
    mo.stop(not start_training_button.value, mo.md("▶️ Press the **'Start / Continue Training'** button above to begin training."))

    def train_one_epoch(epoch: int, n_epochs: int, steps_per_epoch: int, dataloader: DataLoader) -> float:
        """Trains the model for one epoch."""
        model.train()  # Set model to training mode
        epoch_loss = 0.0

        # Use Marimo's progress bar
        with mo.status.progress_bar(total=steps_per_epoch, title=f"Epoch {epoch}/{n_epochs}") as bar:
            for step, batch in enumerate(dataloader):
                if step >= steps_per_epoch:
                    break  # Limit steps per epoch

                optimizer.zero_grad()
                batch = batch.to(device)  # Move data to the correct device
                b_size = batch.shape[0]

                # 1. Sample random timesteps t ~ Uniform({0, ..., T-1})
                t = torch.randint(0, timesteps, (b_size,), device=device).long()

                # 2. Sample noise eps ~ N(0, I)
                noise = torch.randn_like(batch)

                # 3. Calculate noisy image x_t using q_sample (forward process)
                x_noisy = q_sample(x_start=batch, t=t, noise=noise)

                # 4. Predict noise using the U-Net model: eps_theta(x_t, t)
                predicted_noise = model(x_noisy, t)

                # 5. Calculate loss: MSE between actual noise and predicted noise
                loss = loss_fn(noise, predicted_noise)

                # 6. Backpropagate and update weights
                loss.backward()
                # Optional: Gradient Clipping (helps prevent exploding gradients)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                _current_loss = loss.item()
                epoch_loss += _current_loss

                # Update progress bar
                if True:  # (step + 1) % 5 == 0: # Update less frequently for performance
                    _avg_loss_so_far = epoch_loss / (step + 1)

                    bar.update(subtitle=f"Loss: {_current_loss:.4f} | Avg Loss: {_avg_loss_so_far:.4f}")

            # Ensure final update for the bar
            _avg_loss_so_far = epoch_loss / steps_per_epoch

            bar.update(subtitle=f"Avg Loss: {_avg_loss_so_far:.4f}")

        return epoch_loss / steps_per_epoch  # Return average loss for the epoch

    def train(n_epochs: int, current_start_epoch):
        """Main Training Execution for many epoch."""
        total_start_time = time.time()
        training_status = ""
        print(f"Starting training from epoch {current_start_epoch} up to {n_epochs}...")

        # --- Prepare DataLoader (Create it here to use current batch size) ---
        dataloader = get_dataloader()
        actual_steps = min(len(dataloader), steps_per_epoch_slider.value)
        print(f"Using {actual_steps} steps per epoch (DataLoader has {len(dataloader)} batches).")

        try:
            for epoch in range(current_start_epoch, n_epochs + 1):
                epoch_start_time = time.time()
                avg_epoch_loss = train_one_epoch(epoch, n_epochs, actual_steps, dataloader)
                epoch_duration = time.time() - epoch_start_time

                # Log epoch results using Marimo state
                set_epoch_log(get_epoch_log() + [(epoch, avg_epoch_loss)])

                print(f"Epoch {epoch}/{n_epochs} finished. Avg Loss: {avg_epoch_loss:.5f}, Duration: {epoch_duration:.2f}s")

                # Update start epoch for potential interruption/resumption
                set_start_epoch(epoch + 1)

                # --- Checkpointing ---
                if epoch % save_freq_slider.value == 0 or epoch == n_epochs:
                    # Create a config dict to save with the checkpoint
                    chkpt_config = {
                        "T": timesteps,
                        "schedule": noise_schedule_dropdown.value,
                        # Add other relevant settings like model dims if needed
                    }
                    save_checkpoint(model, optimizer, epoch, avg_epoch_loss, chkpt_config)

            training_status = f"✅ Training finished successfully up to epoch {n_epochs}."
            print(training_status)

        except Exception as e:
            training_status = f"❌ Training interrupted at epoch {get_start_epoch() - 1} due to error: {e}"
            print(f"{training_status}\n{traceback.format_exc()}")
            # Save a final checkpoint on error if possible
            last_epoch = get_start_epoch() - 1
            if last_epoch >= current_start_epoch:
                print("Attempting to save checkpoint before exiting...")
                chkpt_config = {"T": timesteps, "schedule": noise_schedule_dropdown.value}
                # Need last loss - could try getting it from log or pass 0
                last_loss = get_epoch_log()[-1][1] if get_epoch_log() else 0.0
                save_checkpoint(model, optimizer, last_epoch, last_loss, chkpt_config)

        _total_duration = time.time() - total_start_time
        print(f"Total training time for this run: {_total_duration:.2f}s")

        # --- Display Final Status ---
        # The loss plot should update automatically via the `display_loss_plot` function in its cell
        return mo.md(f"**Training Run Complete:** {training_status} (Total time: {_total_duration:.2f}s)")

    # Training
    _max_epochs = n_epochs_slider.value
    _current_start_epoch = get_start_epoch()
    if _current_start_epoch > _max_epochs:
        final_status_md = mo.md(f"✅ Training already completed up to epoch {_max_epochs}. No further training needed.").callout("success")
    else:
        final_status_md = train(_max_epochs, _current_start_epoch)

    final_status_md
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## ⚙️ Inference time
        generate some denoise chain: 🪥🪥🪥🖼️
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 🪛 DDPM and DDIM sampling""")
    return


@app.cell(hide_code=True)
def ddpm_sampling(nn, torch):
    # DDPM sampling
    # --- Diffusion Scheduler ---
    # (Keep existing scheduler code: linear_beta_schedule, timesteps, betas, alphas, etc.)
    # timesteps = timesteps_slider.value # timesteps is already defined in the original cell based on the slider
    # ... keep betas, alphas, alphas_cumprod, alphas_cumprod_prev ...
    # ... keep extract function ...
    # ... keep q_sample function ...

    # --- Reverse Process (p - sampling) ---

    # DDPM Sampling Step: p(x_{t-1} | x_t)
    # Uses model prediction epsilon_theta(x_t, t)
    @torch.no_grad()
    def p_sample_ddpm(
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        sqrt_one_minus_alphas_cumprod_t: torch.Tensor, 
        sqrt_recip_alphas_t: torch.Tensor, 
        betas_t: torch.Tensor) -> torch.Tensor:
        """
        Performs one step of the DDPM reverse process (sampling x_{t-1} from x_t).
        Uses the simplified variance sigma_t^2 = beta_t.

        Args:
            model (nn.Module): The model predicting the noise (epsilon_theta).
            x_t (torch.Tensor): The noisy data at timestep t [B, C, H, W].
            t (torch.Tensor): The current timestep tensor [B].
            t_index (int): The integer value of the current timestep.
            sqrt_one_minus_alphas_cumprod_t (torch.Tensor): Pre-extracted sqrt(1 - alpha_cumprod_t).
            sqrt_recip_alphas_t (torch.Tensor): Pre-extracted sqrt(1 / alpha_t).
            betas_t (torch.Tensor): Pre-extracted beta_t.

        Returns:
            torch.Tensor: The sampled data at timestep t-1, x_{t-1} [B, C, H, W].
        """

        # Use model to predict noise added at step t
        predicted_noise = model(x_t, t)

    	# Calculate mean of p(x_{t-1} | x_t) (Equation 11 from DDPM paper)
        # mean = sqrt(1/alpha_t) * (x_t - (beta_t / sqrt(1-alpha_cumprod_t)) * predicted_noise)
        model_mean = sqrt_recip_alphas_t * ( x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            # At the last step (t=0), the output is just the mean
            return model_mean # No noise added at the last step
        else:
            # Calculate variance sigma_t^2 * I. Here we use sigma_t^2 = beta_t.
            # The other option is sigma_t^2 = beta_tilde_t = posterior_variance_t
            posterior_variance_t = betas_t # sigma_t^2 = beta_t
            noise = torch.randn_like(x_t)
            # Algorithm 2 line 4: x_{t-1} = mean + sqrt(variance) * z, where z ~ N(0, I)
            return model_mean + torch.sqrt(posterior_variance_t) * noise




    return (p_sample_ddpm,)


@app.cell(hide_code=True)
def ddim_sampling(Tuple, extract, nn, torch):
    # DDIM sampling

    @torch.no_grad()
    def p_sample_ddim(
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,          # Current timestep tensor [B]
        # t_prev: torch.Tensor,   # Previous timestep tensor [B] - No longer needed directly for indexing alpha_bar_prev
        eta: float,               # Stochasticity parameter (0=deterministic, 1=DDPM-like)
        alphas_cumprod: torch.Tensor, # Full schedule [T] - used for alpha_bar_t
        alphas_cumprod_prev: torch.Tensor, # Full schedule shifted [T] (starts with 1.0) - used for alpha_bar_{t-1}
        sqrt_one_minus_alphas_cumprod: torch.Tensor, # Full schedule [T]
        clip_predicted_x0: bool = True # Whether to clip predicted x0 to [-1, 1] range
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one step of the DDIM reverse process.
        CORRECTED: Uses alphas_cumprod_prev indexed by t to get alpha_bar_{t-1}.

        Args:
            model (nn.Module): The model predicting the noise (epsilon_theta).
            x_t (torch.Tensor): The noisy data at the current timestep t [B, C, H, W].
            t (torch.Tensor): Tensor of current timesteps [B]. Ranges from T-1 down to 0.
            eta (float): Controls the stochasticity (0 for deterministic DDIM).
            alphas_cumprod (torch.Tensor): Pre-calculated cumulative product of alphas [T].
            alphas_cumprod_prev (torch.Tensor): Pre-calculated schedule where index t holds alpha_bar_{t-1}. [T].
            sqrt_one_minus_alphas_cumprod (torch.Tensor): Pre-calculated sqrt(1 - a_bar_t) [T].
            clip_predicted_x0 (bool): If True, clips predicted x0 to [-1, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_{t-1} (torch.Tensor): The sampled data at the previous timestep [B, C, H, W].
                - predicted_x0 (torch.Tensor): The predicted clean data x0 [B, C, H, W].
        """
        # 1. Get schedule constants using the current timestep t
        alpha_bar_t = extract(alphas_cumprod, t, x_t.shape)           # a_bar_t
        # Correctly get a_bar_{t-1} using alphas_cumprod_prev indexed by t
        alpha_bar_prev_t = extract(alphas_cumprod_prev, t, x_t.shape) # a_bar_{t-1}
        sqrt_one_minus_alpha_bar_t = extract(sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # 2. Predict noise using the model
        predicted_noise = model(x_t, t) # epsilon_theta(x_t, t)

        # 3. Predict x0 using the DDIM formula
        # x0_pred = (x_t - sqrt(1 - a_bar_t) * eps_theta) / sqrt(a_bar_t)
        predicted_x0 = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / torch.sqrt(alpha_bar_t)

        # Optional: Clip predicted x0
        if clip_predicted_x0:
            predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)

        # 4. Calculate sigma_t (controls stochasticity)
        # sigma_t = eta * sqrt((1 - a_bar_{t-1}) / (1 - a_bar_t) * (1 - a_bar_t / a_bar_{t-1}))
        # Note: The variance calculation uses alpha_bar_prev_t which we now get correctly.
        term1 = (1.0 - alpha_bar_prev_t) / torch.clamp(1.0 - alpha_bar_t, min=1e-6)
        term2 = 1.0 - alpha_bar_t / torch.clamp(alpha_bar_prev_t, min=1e-6)
        variance = term1 * term2
        variance = torch.clamp(variance, min=1e-20) # Ensure non-negative before sqrt

        sigma_t = eta * torch.sqrt(variance)

        # 5. Calculate the direction pointing to x_t (which is related to the predicted noise)
        # This term is sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * predicted_noise
        # Ensure sqrt argument is non-negative
        sqrt_term = torch.clamp(1.0 - alpha_bar_prev_t - sigma_t**2, min=0.0)
        direction_pointing_to_xt = torch.sqrt(sqrt_term) * predicted_noise

        # 6. Calculate x_{t-1}
        # x_{t-1} = sqrt(a_bar_{t-1}) * x0_pred + direction_pointing_to_xt + sigma_t * noise
        term_x0 = torch.sqrt(alpha_bar_prev_t) * predicted_x0

        # Generate random noise (scaled by sigma_t) only if eta > 0
        # Check t value to avoid adding noise at the very last step (t=0) if desired?
        # DDIM paper implies noise is added at every step where eta > 0.
        # Let's stick to adding noise if eta > 0.
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        term_noise = sigma_t * noise

        # Combine terms
        x_prev = term_x0 + direction_pointing_to_xt + term_noise

        return x_prev, predicted_x0
    return (p_sample_ddim,)


@app.cell(hide_code=True)
def _(mo, sampler_dropdown):
    mo.vstack([mo.md("### Sampling config"),
        sampler_dropdown,
        ])
    return


@app.cell(hide_code=True)
def _(mo, sampler_dropdown, timesteps_slider):
    _max = (2 * timesteps_slider.value) -1
    ddim_steps_slider = mo.ui.slider(1, _max, step=1, value=50, label="Inference Steps")
    ddim_eta_slider = mo.ui.slider(0.0, 1.0, step=0.1, value=0.0, label="Eta (0=Deterministic)")
    mo.hstack(["DDIM specifics:", ddim_steps_slider, ddim_eta_slider], justify="end") if sampler_dropdown.value == "DDIM" else None
    return ddim_eta_slider, ddim_steps_slider


@app.cell(hide_code=True)
def _(generate_samples_button, mo, num_samples_slider):
    mo.hstack([generate_samples_button, num_samples_slider], justify="start")
    return


@app.cell(hide_code=True)
def sampling_loop(
    List,
    T,
    Tuple,
    alphas_cumprod: "torch.Tensor",
    alphas_cumprod_prev: "torch.Tensor",
    betas,
    channels,
    ddim_eta_slider,
    ddim_steps_slider,
    device,
    extract,
    generate_samples_button,
    image_size_slider,
    make_grid,
    math,
    mo,
    model,
    nn,
    np,
    num_samples_slider,
    p_sample_ddim,
    p_sample_ddpm,
    sampler_dropdown,
    sqrt_one_minus_alphas_cumprod: "torch.Tensor",
    sqrt_recip_alphas: "torch.Tensor",
    timesteps_slider,
    torch,
    traceback,
):
    num_display_steps = 15 # How many steps (including final but not initial) noise image to show

    # DDPM Sampling Loop (MODIFIED)
    @torch.no_grad()
    def p_sample_loop_ddpm(
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        totalTimeSteps: int,
        num_display_steps: int = 10
    ) -> List[torch.Tensor]:
        """
        Runs the full DDPM reverse sampling loop (Algorithm 2).

        Args:
            model (nn.Module): The trained noise prediction model.
            shape (Tuple[int, ...]): The desired output shape [B, C, H, W].
            device (torch.device): The device to perform sampling on (e.g., 'cuda', 'cpu').
            totalTimeSteps (int): total number of timesteps
            num_display_steps (int): How many intermediate steps (including start/end) to store.

        Returns:
            List[torch.Tensor]: A list of tensors representing the denoising process,
                                from initial noise to the final image. Stored on CPU.
        """
        b = shape[0]
        # Start with random noise N(0, I) at timestep T
        img = torch.randn(shape, device=device)
        imgs_over_time = [img.cpu()] # Store initial noise

        # Calculate interval for storing intermediate images
        if num_display_steps <= 1:
            store_interval = totalTimeSteps + 1 # Only store start/end if steps <= 1
        else:
            # Ensure we capture steps near the beginning and the very end
            store_interval = max(1, totalTimeSteps // (num_display_steps - 1))

        # Iterate from T-1 down to 0
        steps_to_process = list(reversed(range(0, totalTimeSteps)))

        with mo.status.progress_bar(total=totalTimeSteps, title="DDPM sampling") as bar:
            for i, t_index in enumerate(steps_to_process):
                # Create a tensor for the current timestep t for the whole batch
                t = torch.full((b,), t_index, device=device, dtype=torch.long)

                # Pre-extract constants needed for p_sample for this specific timestep t
                # These constants depend on the schedule chosen initially (Linear/Cosine)
                _betas_t = extract(betas, t, img.shape)
                _sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, img.shape)
                _sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, img.shape)

                # Perform one sampling step
                img = p_sample_ddpm(
                    model,
                    img, # Current image x_t
                    t, # Current timestep tensor
                    t_index, # Current timestep integer value
                    _sqrt_one_minus_alphas_cumprod_t,
                    _sqrt_recip_alphas_t,
                    _betas_t
                )

                # Store image at intervals OR if it's the last step (t_index == 0)
                is_last_step = (t_index == 0)
                is_store_interval = ((i + 1) % store_interval == 0)

                if is_store_interval or is_last_step:
                     # Avoid duplicates if last step aligns with interval
                     if not imgs_over_time or not torch.equal(imgs_over_time[-1], img.cpu()):
                         imgs_over_time.append(img.cpu()) # Store intermediate/final image on CPU
                bar.update(subtitle=f"i={i} t_index={t_index}")

        # Ensure the very last computed image (t=0) is stored if somehow missed by interval logic
        # This check might be redundant with the is_last_step check inside the loop, but safe to keep.
        if not torch.equal(imgs_over_time[-1], img.cpu()):
            imgs_over_time.append(img.cpu())

        return imgs_over_time

    # --- DDIM Sampling Loop ---

    @torch.no_grad()
    def p_sample_loop_ddim(
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        totalTimeSteps: int,
        num_inference_steps: int,      # Number of steps for DDIM sampling (can be < T)
        eta: float = 0.0,              # DDIM eta parameter (0 = deterministic)
        num_display_steps: int = 10,   # How many steps to store for visualization
        clip_predicted_x0: bool = True # Passed to p_sample_ddim
    ) -> List[torch.Tensor]:
        """
        Runs the full DDIM reverse sampling loop.

        Args:
            model (nn.Module): The trained noise prediction model.
            shape (Tuple[int, ...]): The desired output shape [B, C, H, W].
            device (torch.device): The device to perform sampling on.
            num_inference_steps (int): The number of DDIM steps to perform.
            eta (float): Controls stochasticity (0=deterministic, 1=DDPM-like).
            num_display_steps (int): Approximate number of intermediate steps to store.
            clip_predicted_x0 (bool): Whether to clip predicted x0 in p_sample_ddim.

        Returns:
            List[torch.Tensor]: A list of tensors representing the denoising process,
                                from initial noise to the final image. Stored on CPU.
        """
        # 1. Define the timestep sequence for DDIM
        # We sample 'num_inference_steps' steps from the original T steps
        # Typically spaced evenly on a linear or squared scale
        step_ratio = totalTimeSteps / num_inference_steps
        timesteps_np = (np.arange(0, num_inference_steps) * step_ratio).round().astype(np.int64)
        timesteps_np = timesteps_np[::-1].copy() # Reverse order: T-like -> 0
        timesteps = torch.from_numpy(timesteps_np).to(device)

        # Note: t_prev is not strictly needed inside p_sample_ddim anymore for indexing,
        # but the sequence generation logic remains the same.
        # We don't need to compute prev_timesteps_tensor if it's not used.

        # 2. Initialize starting noise
        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs_over_time = [img.cpu()] # Store initial noise

        # 3. Calculate display intervals
        if num_display_steps <= 1:
            store_interval = num_inference_steps + 1 # Only store start/end
        else:
            store_interval = max(1, num_inference_steps // (num_display_steps - 1))

        # 4. DDIM sampling loop
        with mo.status.progress_bar(total=num_inference_steps, title="DDIM sampling") as bar:
            for i in range(num_inference_steps):
                t_index = i # Index in our custom DDIM sequence
                t_val = timesteps[t_index] # Current timestep value

                # Create batch tensor for the current timestep
                t_batch = torch.full((b,), t_val.item(), device=device, dtype=torch.long)

                # Call the DDIM step function (passing the correct alpha schedules)
                img, _ = p_sample_ddim( # We don't store predicted_x0 in this loop
                    model,
                    img,          # x_t
                    t_batch,      # Current timestep tensor
                    # t_prev_batch no longer needed in signature
                    eta,
                    alphas_cumprod, # Pass the full schedules
                    alphas_cumprod_prev,
                    sqrt_one_minus_alphas_cumprod,
                    clip_predicted_x0=clip_predicted_x0
                )

                # Store image at intervals or if it's the last step
                is_last_step = (i == num_inference_steps - 1)
                # Ensure display steps are calculated correctly relative to num_inference_steps
                is_store_interval = (num_display_steps > 1) and ((i + 1) % store_interval == 0)

                # Store if it's an interval step OR the very last step
                if is_store_interval or is_last_step:
                     # Avoid storing duplicates if last step aligns with interval
                     if not imgs_over_time or not torch.equal(imgs_over_time[-1], img.cpu()):
                         imgs_over_time.append(img.cpu())
                bar.update()

        # Ensure the absolutely final image is stored if the loop logic missed it
        if not torch.equal(imgs_over_time[-1], img.cpu()):
            imgs_over_time.append(img.cpu())

        return imgs_over_time

    # === Helper Function: Tensor to PIL Grid ===
    # Inverse transform: [-1, 1] -> [0, 1] -> [0, 255] -> PIL Image
    def tensor_batch_to_pil_grid(img_tensor_batch, num_samples):
        # Ensure tensor is on CPU before grid/PIL conversion
        img_tensor_batch = img_tensor_batch.cpu()
        img_tensor_batch = (img_tensor_batch + 1) / 2 # [-1, 1] -> [0, 1]
        img_tensor_batch = img_tensor_batch.clamp(0, 1) # Ensure range

        # Create a grid
        grid = make_grid(img_tensor_batch, nrow=int(math.sqrt(num_samples)), padding=5)
        pil_img = T.ToPILImage()(grid)
        return pil_img

    # === Marimo State for Visualization ===
    # Stores the list of image batches from the denoising process
    get_denoising_steps, set_denoising_steps = mo.state([])
    # Stores the index of the step currently being viewed
    get_current_step_index, set_current_step_index = mo.state(0)
    # Status message area
    get_sampling_status, set_sampling_status = mo.state(mo.md("Click 'Generate Samples' to visualize the denoising process."))

    # === Generation Logic (Triggered by Button) ===
    # if the button hasn't been clicked, don't run.
    mo.stop(not generate_samples_button.value, mo.md("Press 'Generate Samples' button to run 🔥"))



    def sampling(num_samples: int, image_size: int, sampler: str):
        status = mo.md(f"Generating {num_samples} samples...")
        model.eval() # Set model to evaluation mode

        sample_shape = (num_samples, channels, image_size, image_size)

        # Get the current diffusion timestep count T from the UI slider
        totalTimeSteps = timesteps_slider.value # Use the value from the config slider
        generated_batches = []
        try:
            # --- Select Sampler and Generate ---
            if sampler_dropdown.value == "DDPM":
                print(f"Generating {num_samples} samples using {sampler} (T={totalTimeSteps}). Storing ~{num_display_steps} steps...")
                generated_batches = p_sample_loop_ddpm(model, sample_shape, device, totalTimeSteps, num_display_steps)

            elif sampler_dropdown.value == "DDIM":
                _num_inference_steps = ddim_steps_slider.value
                _eta = ddim_eta_slider.value
                print(f"Generating {num_samples} samples using {sampler} (Steps={_num_inference_steps} from {totalTimeSteps}, Eta={_eta}). Storing ~{num_display_steps} steps...")
                generated_batches = p_sample_loop_ddim(model, sample_shape, device, totalTimeSteps, _num_inference_steps, _eta, num_display_steps)
            else:
                 generated_batches = [] # Clear any previous results

            # --- Update State After Generation ---
            if generated_batches:
                set_denoising_steps(generated_batches)
                set_current_step_index(0) # Reset view to the start (noise)            
                status = f"Generated {num_samples} samples using {sampler}."
            else:
                 # Handle case where generation failed or returned empty
                 set_denoising_steps([])

                 if sampler in ["DDPM", "DDIM"]: # Only show error if valid sampler failed
                     status = "⚠️ Sample generation failed or returned no results."
                 else:
                     status = "🚨 Invalid sampler selected."

        except Exception as e:
            status = f"🚨Error during sampling: {e}\n{traceback.format_exc()}"
            set_denoising_steps([])
        return status

    set_sampling_status(sampling(num_samples_slider.value, image_size_slider.value, sampler_dropdown.value))
    return (
        get_denoising_steps,
        get_sampling_status,
        num_display_steps,
        tensor_batch_to_pil_grid,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 📽️ Display time""")
    return


@app.cell(hide_code=True)
def _(get_sampling_status, mo, num_display_steps):
    step_slider = mo.ui.slider(
                start=1,
                stop=num_display_steps+1,
                value=1,
                step=1,
                label="Select the denoising step to be displayed"
            )

    mo.vstack([step_slider,get_sampling_status()])
    return (step_slider,)


@app.cell(hide_code=True)
def _(
    BytesIO,
    get_denoising_steps,
    mo,
    num_samples_slider,
    step_slider,
    tensor_batch_to_pil_grid,
    traceback,
):
    # === Display Logic (Reacts to State Changes) ===
    def build_image(denoising_steps, viewing_index: int, num_samples: int):
        try:       
            result = mo.md(f"Nothing yet {viewing_index}").callout("info")
            # Prepare the image display area
            result = mo.md("No steps generated yet.")
            if denoising_steps:
                num_actual_steps = len(denoising_steps)
                if 0 <= viewing_index < num_actual_steps:
                    # Get the tensor batch for the selected step
                    current_batch = denoising_steps[viewing_index]

                    # Convert to PIL grid
                    pil_grid = tensor_batch_to_pil_grid(current_batch, num_samples)

                    # Convert PIL image to bytes for display
                    buf = BytesIO()
                    pil_grid.save(buf, format='PNG')
                    buf.seek(0)

                    # Determine step label (e.g., "Initial Noise", "Step X", "Final Image")
                    step_label = ""
                    if viewing_index == 0:
                        step_label = "(Initial Noise)"
                    elif viewing_index == num_actual_steps - 1:
                        step_label = "(Final Image)"

                    result = mo.vstack([
                        mo.md(f"**Step {viewing_index + 1} / {num_actual_steps}** {step_label}"),
                        mo.image(buf.getvalue(), height=500)
                    ])
                else:
                    # Handle invalid index (shouldn't happen with slider constraints)
                    result = mo.md(f"⚠️ Invalid step index selected: {viewing_index}")

        except Exception as e:
            err_msg = f"Error during sampling: {e}\n{traceback.format_exc()}"
            print(err_msg)
            result = mo.md(err_msg).callout("danger")


        return result

    # Combine status, slider (if available), and image display
    mo.vstack([
        build_image(get_denoising_steps(), step_slider.value -1, num_samples_slider.value)
    ])
    return


if __name__ == "__main__":
    app.run()
