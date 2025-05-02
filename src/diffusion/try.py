

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
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
def _(mo):
    # --- General Settings ---
    image_size_slider = mo.ui.slider(
        128, 128, value=128, label="Image Size (Fixed)"
    )
    batch_size_slider = mo.ui.slider(
        2, 32, step=2, value=4, label="Batch Size"
    )
    n_epochs_slider = mo.ui.slider(
        1, 100, step=1, value=10, label="Epochs"
    )
    steps_per_epoch_slider = mo.ui.slider(
        50, 500, step=10, value=100, label="Steps per Epoch"
    )
    learning_rate_slider = mo.ui.number(
        1e-5, 1e-2, step=1e-5, value=5e-4, label="Learning Rate"
    )

    # --- Diffusion Settings ---
    timesteps_slider = mo.ui.slider(
        100, 1000, step=50, value=300, label="Diffusion Timesteps (T)"
    )
    noise_schedule_dropdown = mo.ui.dropdown(
        ["Linear", "Cosine"], value="Cosine", label="Noise Schedule"
    )

    # --- Data Generation Settings ---
    color_mode_dropdown = mo.ui.dropdown(
        ["Grayscale", "RGB"], value="RGB", label="Image Mode"
    )
    num_solids_range_slider = mo.ui.range_slider(
        1, 8, value=(2, 5), label="Number of Solids per Image"
    )

    # --- Model & Checkpointing ---
    checkpoint_dir_text = mo.ui.text(
        "models/diffusion_platonic", label="Directory"
    )
    load_checkpoint_checkbox = mo.ui.checkbox(
        value=True, label="Load Latest if Available"
    )
    save_freq_slider = mo.ui.slider(
        1, 10, value=2, label="Save Every (Epochs)"
    )

    # --- Sampling Controls ---
    sampler_dropdown = mo.ui.dropdown(
        ["DDPM", "DDIM"], value="DDIM", label="Sampler Type"
    )
    ddim_steps_slider = mo.ui.slider(
        10, 100, step=5, value=50, label="Inference Steps"
    )
    ddim_eta_slider = mo.ui.slider(
        0.0, 1.0, step=0.1, value=0.0, label="Eta (0=Deterministic)"
    )
    num_samples_slider = mo.ui.slider(
        1, 16, value=4, label="Samples to Generate"
    )
    num_display_steps_slider = mo.ui.slider(
        4, 20, value=12, label="Denoising Steps to Visualize"
    )

    # --- UI Buttons ---
    generate_data_preview_button = mo.ui.button(label="Preview Generated Data")
    start_training_button = mo.ui.run_button(label="Start / Continue Training")
    generate_samples_button = mo.ui.run_button(label="Generate Samples")

    # --- Layout ---
    ui_config_layout = mo.vstack([
        mo.md("### General & Training"),
        mo.hstack([learning_rate_slider,image_size_slider]),
        mo.hstack([
            batch_size_slider, n_epochs_slider, steps_per_epoch_slider,
        ]),
        mo.md("### Diffusion Process"),
        mo.hstack([noise_schedule_dropdown, timesteps_slider], justify="start"),
        mo.md("### Data Generation"),
        mo.hstack([color_mode_dropdown, num_solids_range_slider]),
        mo.md("### Model & Checkpointing"),
        mo.hstack([checkpoint_dir_text, save_freq_slider, load_checkpoint_checkbox]),
        mo.md("### Sampling"),
        sampler_dropdown,
        mo.hstack([
            "DDIM specifics:",
            ddim_steps_slider,
            ddim_eta_slider
        ], justify="end"),
        mo.hstack([
            num_samples_slider,
            num_display_steps_slider,
        ]),
        mo.md("---"),
        mo.hstack([
            generate_data_preview_button,
            start_training_button,
            generate_samples_button,
        ], justify="start"),
    ])
    return (
        batch_size_slider,
        checkpoint_dir_text,
        color_mode_dropdown,
        ddim_eta_slider,
        ddim_steps_slider,
        generate_data_preview_button,
        image_size_slider,
        learning_rate_slider,
        load_checkpoint_checkbox,
        n_epochs_slider,
        noise_schedule_dropdown,
        num_display_steps_slider,
        num_samples_slider,
        num_solids_range_slider,
        sampler_dropdown,
        save_freq_slider,
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
    ddim_eta_slider,
    ddim_steps_slider,
    image_size_slider,
    learning_rate_slider,
    load_checkpoint_checkbox,
    mo,
    n_epochs_slider,
    noise_schedule_dropdown,
    num_display_steps_slider,
    num_samples_slider,
    num_solids_range_slider,
    sampler_dropdown,
    save_freq_slider,
    steps_per_epoch_slider,
    timesteps_slider,
):
    mo.md(f"""
    **Current Configuration Summary:**  
     - Image: {image_size_slider.value}x{image_size_slider.value}, Mode: {color_mode_dropdown.value}  
     - Training: {n_epochs_slider.value} epochs, {steps_per_epoch_slider.value} steps/epoch, Batch: {batch_size_slider.value}, LR: {learning_rate_slider.value:.1e}  
     - Diffusion: T={timesteps_slider.value}, Schedule: {noise_schedule_dropdown.value}  
     - Data: {num_solids_range_slider.value[0]}-{num_solids_range_slider.value[1]} solids/image  
     - Checkpointing: Dir: `{checkpoint_dir_text.value}`, Save every {save_freq_slider.value} epochs, Load: {load_checkpoint_checkbox.value}  
     - Sampling: Sampler: {sampler_dropdown.value}, Num Samples: {num_samples_slider.value}, Visualize Steps: {num_display_steps_slider.value}  
     - DDIM Specific: Steps: {ddim_steps_slider.value}, Eta: {ddim_eta_slider.value}  
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🧱 Synthetic Data Generation""")
    return


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
    math,
    mo,
    np,
    random,
):
    # --- Platonic Solid Definitions (Simplified 2D Projections) ---
    PLATONIC_SOLIDS: Dict[str, List[Tuple[float, float]]] = {
        "Tetrahedron": [(0.0, -0.8), (0.7, 0.4), (-0.7, 0.4)],
        "Cube": [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
        "Octahedron": [(0.0, -0.8), (0.7, 0.0), (0.0, 0.8), (-0.7, 0.0)],
        "Dodecahedron": [
            (0.0, -0.8), (0.76, -0.25), (0.47, 0.65), (-0.47, 0.65), (-0.76, -0.25),
        ],
        "Icosahedron": [
            (0.4, -0.7), (0.8, 0.0), (0.4, 0.7), (-0.4, 0.7), (-0.8, 0.0), (-0.4, -0.7),
        ],
    }
    SOLID_NAMES: List[str] = list(PLATONIC_SOLIDS.keys())

    # --- Colors ---
    VIBRANT_COLORS: Dict[str, Tuple[int, int, int]] = {
        "Tetrahedron": (255, 69, 0),    # OrangeRed
        "Cube": (30, 144, 255),   # DodgerBlue
        "Octahedron": (50, 205, 50),    # LimeGreen
        "Dodecahedron": (255, 215, 0),   # Gold
        "Icosahedron": (153, 50, 204),   # DarkOrchid
    }
    GRAY_COLORS: Dict[str, Tuple[int, int, int]] = {
        name: (random.randint(100, 200),) * 3 for name in SOLID_NAMES
    }

    def _get_solid_vertices(
        solid_name: str, center_x: float, center_y: float, size: float
    ) -> List[Tuple[float, float]]:
        """Scales and translates predefined solid vertices."""
        _vertices = PLATONIC_SOLIDS[solid_name]
        _scaled_vertices = [
            (center_x + x * size, center_y + y * size) for x, y in _vertices
        ]
        return _scaled_vertices

    def generate_scene_image(
        img_size: int = 128,
        mode: str = "RGB",
        num_solids_min: int = 2,
        num_solids_max: int = 5,
    ) -> Image.Image:
        """Generates a PIL image with random Platonic solids."""
        _final_mode = "L" if mode == "Grayscale" else "RGB"
        _num_channels = 1 if _final_mode == "L" else 3

        if _final_mode == "L":
            _background_color: Union[int, Tuple[int, int, int]] = random.randint(30, 70)
        else:
            _background_color = (
                random.randint(30, 50), random.randint(30, 50), random.randint(50, 70)
            )

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
                    _dist = math.sqrt((_center_x - _px)**2 + (_center_y - _py)**2)
                    if _dist < (_solid_size + _psize) * 0.6: # Check distance
                        _too_close = True
                        break
                if not _too_close:
                    _placed = True
                    break

            if not _placed:
                continue # Skip if placement failed

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
            _shadow_layer = Image.new(_final_mode, image.size, 0) # Transparent if RGBA, black if L/RGB
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
                _mask = _shadow_layer # In L mode, this works directly. In RGB, it's trickier.
                if _final_mode == 'RGB':
                    # Create a proper alpha mask from the shadow layer for RGB
                     _mask = Image.new("L", image.size, 0)
                     _intensity_threshold = 10 # Only blend where shadow is somewhat visible
                     _shadow_data = np.array(_shadow_layer)
                     _mask_data = np.where(np.mean(_shadow_data, axis=2) > _intensity_threshold, 255, 0).astype(np.uint8)
                     _mask = Image.fromarray(_mask_data)


                _shadow_color_img = Image.new(_final_mode, image.size, _shadow_color)
                image.paste(_shadow_color_img, (0, 0), mask=_mask)
            except Exception:
                # Fallback if masking fails: just draw the polygon
                draw.polygon(_shadow_vertices, fill=_shadow_color)


            # --- Draw Solid ---
            draw.polygon(_vertices, fill=_solid_color, outline=_outline_color, width=1) # Added outline width
            _placed_solids.append((_solid_name, _center_x, _center_y, _solid_size))

        return image

    def display_sample_images(
        num: int = 4,
        img_size: int = 128,
        mode: str = "RGB",
        num_solids_min: int = 2,
        num_solids_max: int = 5,
    ) -> mo.Html:
        """Generates and displays a grid of sample images."""
        images = [
            generate_scene_image(img_size, mode, num_solids_min, num_solids_max)
            for _ in range(num)
        ]
        byte_streams = []
        for img in images:
            _buf = BytesIO()
            img.save(_buf, format="PNG")
            _buf.seek(0)
            byte_streams.append(_buf.getvalue())

        _num_cols = int(math.sqrt(num))
        return mo.hstack([mo.image(bs) for bs in byte_streams], justify="center")

    print("Data generation functions defined.")
    return display_sample_images, generate_scene_image


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
    num_solids_range_slider,
):
    mo.stop(generate_data_preview_button.value, mo.md("Click 'Preview Generated Data' to see examples."))

    # Button clicked, generate and display samples
    display_sample_images(
        num=4,
        img_size=image_size_slider.value,
        mode=color_mode_dropdown.value,
        num_solids_min=num_solids_range_slider.value[0],
        num_solids_max=num_solids_range_slider.value[1],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🎞️ Dataset and DataLoader""")
    return


@app.cell(hide_code=True)
def _(
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
    num_solids_range_slider,
    steps_per_epoch_slider,
    torch,
    traceback,
):
    """ Cell for defining on the fly Dataloaders"""

    channels = 3 if color_mode_dropdown.value == "RGB" else 1
    _img_size = image_size_slider.value

    # Define the transformation pipeline
    transform = T.Compose([
        T.ToTensor(),       # PIL Image [0, 255] (H, W, C) or (H,W) -> Torch Tensor [0, 1] (C, H, W)
        T.Lambda(lambda t: (t * 2) - 1) # Scale [0, 1] -> [-1, 1]
    ])

    class SyntheticSolidsDataset(Dataset):
        """A dataset that generates images on-the-fly."""
        def __init__(
            self,
            img_size: int = 128,
            mode: str = "RGB",
            dataset_size: int = 1000, # Virtual size
            transform: Optional[Callable] = None,
            num_solids_min: int = 2,
            num_solids_max: int = 5,
        ):
            self.img_size = img_size
            self.mode = mode
            self.dataset_size = dataset_size
            self.transform = transform
            self.num_solids_min = num_solids_min
            self.num_solids_max = num_solids_max
            self.expected_channels = 1 if self.mode == "Grayscale" else 3
            print(f"Dataset initialized: size={dataset_size}, mode={mode}, solids={num_solids_min}-{num_solids_max}")

        def __len__(self) -> int:
            return self.dataset_size

        def __getitem__(self, idx: int) -> torch.Tensor:
            # Generate image dynamically
            try:
                img = generate_scene_image(
                    self.img_size, self.mode, self.num_solids_min, self.num_solids_max
                )

                if self.transform:
                    img = self.transform(img)

                # Ensure correct channel dimension for grayscale after ToTensor
                if self.mode == "Grayscale" and img.dim() == 2:
                    img = img.unsqueeze(0) # (H, W) -> (1, H, W)

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
        transform=transform,
        num_solids_min=num_solids_range_slider.value[0],
        num_solids_max=num_solids_range_slider.value[1],
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
            drop_last=True, # Important for consistent batch sizes during training
            pin_memory=False # Usually False with num_workers=0
        )

    # --- Test DataLoader ---
    _data_loader_status = ""
    _batch_shape = None
    try:
        _temp_loader = get_dataloader()
        _sample_batch = next(iter(_temp_loader))
        _batch_shape = _sample_batch.shape
        _data_loader_status = f"DataLoader ready. Sample batch shape: `{_batch_shape}`"
        del _temp_loader, _sample_batch # Clean up memory
    except Exception as e:
        _data_loader_status = f"Error creating DataLoader or getting batch: {e}\n{traceback.format_exc()}"
        print(_data_loader_status) # Also print error to console

    dataloader_info = mo.vstack([
        mo.md("Dataset and DataLoader configured for on-the-fly generation."),
        mo.md(f"**DataLoader Status:** {_data_loader_status}")
    ])
    dataloader_info
    return channels, dataloader_info, get_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🧠 Diffusion Model (U-Net)""")
    return


@app.cell(hide_code=True)
def unet_model(Optional, TF, Tuple, math, mo, nn, torch, traceback):
    """ Cell to define the U-Net model """
    # --- Time Embedding ---
    class SinusoidalPosEmb(nn.Module):
        """ Sinusoidal Position Embedding Layer """
        def __init__(self, dim: int):
            super().__init__()
            if dim % 2 != 0:
                raise ValueError("SinusoidalPosEmb dimension must be even.")
            self.dim = dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _device = x.device
            _half_dim = self.dim // 2
            _emb = math.log(10000) / (_half_dim - 1)
            _emb = torch.exp(torch.arange(_half_dim, device=_device) * -_emb)
            _emb = x[:, None] * _emb[None, :]
            _emb = torch.cat((_emb.sin(), _emb.cos()), dim=-1)
            return _emb

    # --- Building Block: Residual Block with Time Embedding ---
    class ResidualBlock(nn.Module):
        """ Residual Block with GroupNorm, SiLU, Conv, Dropout, and optional Time Embedding """
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_emb_dim: Optional[int] = None,
            dropout: float = 0.1,
            groups: int = 8 # GroupNorm groups
        ):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.norm1 = nn.GroupNorm(groups, out_channels)
            self.act1 = nn.SiLU() # Swish activation

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
                self.time_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_emb_dim, out_channels)
                )
            else:
                self.time_mlp = None

        def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
            _h = self.act1(self.norm1(self.conv1(x)))

            # Add time embedding if provided
            if self.time_mlp is not None and time_emb is not None:
                _time_encoding = self.time_mlp(time_emb)
                # Add embedding to feature map channels (broadcast H, W)
                _h = _h + _time_encoding.unsqueeze(-1).unsqueeze(-1)

            _h = self.dropout(self.act2(self.norm2(self.conv2(_h))))

            # Apply skip connection
            return _h + self.skip_connection(x)

    # --- U-Net Architecture ---
    class UNet(nn.Module):
        """ U-Net architecture for noise prediction in Diffusion Models """
        def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            time_emb_dim: int = 128,
            base_dim: int = 32,         # Initial number of channels
            dim_mults: Tuple[int, ...] = (1, 2, 4), # Channel multipliers per resolution level
            dropout: float = 0.1,
            num_res_blocks: int = 2,    # Number of residual blocks per level
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.base_dim = base_dim
            self.dim_mults = dim_mults
            _current_dim = base_dim
            _num_resolutions = len(dim_mults)

            # 1. Time Embedding Projection
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.SiLU(),
                nn.Linear(time_emb_dim * 4, time_emb_dim),
            )
            print(f"UNet Init: Time MLP output dim: {time_emb_dim}")

            # 2. Initial Convolution
            self.init_conv = nn.Conv2d(in_channels, _current_dim, kernel_size=3, padding=1)
            print(f"UNet Init: Initial Conv: {in_channels} -> {_current_dim}")

            # 3. Encoder Path (Downsampling)
            self.downs = nn.ModuleList([])
            _encoder_dims = [_current_dim] # Store output dims of each encoder level
            print("--- UNet Init: Encoder ---")
            for i, mult in enumerate(dim_mults):
                _is_last_level = (i == _num_resolutions - 1)
                _dim_out = base_dim * mult
                _blocks = []
                _blocks.append(ResidualBlock(_current_dim, _dim_out, time_emb_dim, dropout))
                _current_dim = _dim_out
                for _ in range(num_res_blocks - 1):
                     _blocks.append(ResidualBlock(_current_dim, _current_dim, time_emb_dim, dropout))

                # Downsampling layer (except for the last encoder level)
                if not _is_last_level:
                     _blocks.append(nn.Conv2d(_current_dim, _current_dim, kernel_size=3, stride=2, padding=1))
                     print(f"  Level {i}: {_encoder_dims[-1]} -> {_current_dim} channels, Downsample")
                else:
                     _blocks.append(nn.Identity()) # No downsampling at last level
                     print(f"  Level {i}: {_encoder_dims[-1]} -> {_current_dim} channels, No Downsample")

                self.downs.append(nn.ModuleList(_blocks))
                _encoder_dims.append(_current_dim)


            # 4. Bottleneck
            print(f"--- UNet Init: Bottleneck ({_current_dim} channels) ---")
            self.mid_block1 = ResidualBlock(_current_dim, _current_dim, time_emb_dim, dropout)
            self.mid_block2 = ResidualBlock(_current_dim, _current_dim, time_emb_dim, dropout)
            _bottleneck_dim = _current_dim # Save for decoder start

            # 5. Decoder Path (Upsampling)
            self.ups = nn.ModuleList([])
            print("--- UNet Init: Decoder ---")
            # Iterate through encoder levels in reverse, skipping the final output dimension
            for i, mult in reversed(list(enumerate(dim_mults))):
                _is_first_level = (i == 0)
                _dim_in = _current_dim  # Dimension from previous decoder level or bottleneck
                _dim_skip = _encoder_dims[i] # Dimension of skip connection from encoder level i
                _dim_out = base_dim * dim_mults[i-1] if not _is_first_level else base_dim # Target output dim for this level

                _block_input_dim = _dim_in + _dim_skip
                _blocks = []

                # Residual blocks for this level
                _blocks.append(ResidualBlock(_block_input_dim, _current_dim, time_emb_dim, dropout))
                for _ in range(num_res_blocks - 1):
                     _blocks.append(ResidualBlock(_current_dim, _current_dim, time_emb_dim, dropout))

                 # Upsampling layer (except for the first decoder level connecting to bottleneck)
                if i != _num_resolutions -1 : # If not connecting directly to bottleneck
                     _upsample_layer = nn.ConvTranspose2d(_dim_in, _dim_in, kernel_size=4, stride=2, padding=1)
                     print(f"  Level {i}: Upsample {_dim_in} -> {_dim_in}, Input {_block_input_dim} -> {_current_dim}")
                     _blocks.append(_upsample_layer) # Add upsampling layer
                else:
                     print(f"  Level {i}: No Upsample, Input {_block_input_dim} -> {_current_dim}")
                     _blocks.append(nn.Identity()) # No upsampling needed from bottleneck if same res


                self.ups.append(nn.ModuleList(_blocks))
                _current_dim = _dim_out # Update current_dim for the next (higher res) level input


            # 6. Final Convolution Layer
            self.final_conv = nn.Sequential(
                nn.GroupNorm(8, base_dim), # Norm before final conv
                nn.SiLU(),
                nn.Conv2d(base_dim, out_channels, kernel_size=1)
            )
            print(f"--- UNet Init: Final Conv ({base_dim} -> {out_channels}) ---")
            print("--- UNet Init: Complete ---")

        def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            # 1. Get time embedding
            _t_emb = self.time_mlp(time)

            # 2. Initial convolution
            _h = self.init_conv(x) # (B, base_dim, H, W)
            _skips = [_h] # Store for skip connections (after init_conv)

            # 3. Encoder path
            for i, level_blocks in enumerate(self.downs):
                for block in level_blocks[:-1]: # Apply ResBlocks
                    _h = block(_h, _t_emb)
                _skips.append(_h) # Store output *before* downsampling for skip connection
                _downsample_layer = level_blocks[-1] # Get the downsample/identity layer
                _h = _downsample_layer(_h)


            # 4. Bottleneck
            _h = self.mid_block1(_h, _t_emb)
            _h = self.mid_block2(_h, _t_emb)


            # 5. Decoder path
            for i, level_blocks in enumerate(self.ups):
                _res_blocks = level_blocks[:-1] # Get ResBlocks
                _upsample_layer = level_blocks[-1] # Get the upsample/identity layer

                _h = _upsample_layer(_h) # Upsample first

                _skip_connection = _skips.pop()

                # Pad _h if spatial dimensions don't match skip_connection after upsampling
                _diffY = _skip_connection.size()[2] - _h.size()[2]
                _diffX = _skip_connection.size()[3] - _h.size()[3]
                if _diffX != 0 or _diffY != 0:
                    _h = TF.pad(_h, [_diffX // 2, _diffX - _diffX // 2,
                                    _diffY // 2, _diffY - _diffY // 2])

                # Concatenate upsampled tensor and skip connection
                _h = torch.cat((_h, _skip_connection), dim=1)

                # Apply Residual Blocks for this level
                for block in _res_blocks:
                    _h = block(_h, _t_emb)


            # 6. Final convolution
            _out = self.final_conv(_h)
            return _out

    # === Code to Instantiate and Test (Lightweight) ===
    _model_status = "Model definition ready."
    try:
        # Simple test case
        _test_unet = UNet(in_channels=3, out_channels=3, base_dim=16, dim_mults=(1, 2))
        _test_params = sum(p.numel() for p in _test_unet.parameters())
        _model_status = f"U-Net class defined. Test instance (base_dim=16, mults=(1,2)) has {_test_params:,} parameters."
        del _test_unet # clean up
    except Exception as e:
        _model_status = f"Error during U-Net class definition or test: {e}"
        print(f"\nERROR: {_model_status}\n{traceback.format_exc()}")

    model_info = mo.vstack([
        mo.md("U-Net model components (`SinusoidalPosEmb`, `ResidualBlock`, `UNet`) defined."),
        mo.md(f"**Status:** {_model_status}")
    ])
    model_info
    return (UNet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🍵 Diffusion Process Utilities (Noise Schedules & Sampling Steps)""")
    return


@app.cell
def diffusion_tools(
    Optional,
    Tuple,
    mo,
    noise_schedule_dropdown,
    timesteps_slider,
    torch,
):
    """ Cell to store diffusion utilities methods for DDPM and DDIM"""

    # --- Noise Schedule Functions ---
    def linear_beta_schedule(timesteps: int) -> torch.Tensor:
        """ Standard linear schedule from DDPM paper. """
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """ Cosine schedule from 'Improved Denoising Diffusion Probabilistic Models'. """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
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

    # --- Pre-calculate Diffusion Constants ---
    alphas: torch.Tensor = 1.0 - betas
    alphas_cumprod: torch.Tensor = torch.cumprod(alphas, axis=0)
    # Pad with 1.0 at the beginning for calculation at t=0
    alphas_cumprod_prev: torch.Tensor = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]])

    # Required for q_sample (forward process)
    sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(1.0 - alphas_cumprod)

    # Required for DDPM p_sample (reverse process)
    sqrt_recip_alphas: torch.Tensor = torch.sqrt(1.0 / alphas)
    posterior_variance: torch.Tensor = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) # beta_tilde_t

    print(f"Diffusion constants calculated for T={timesteps} using {_schedule_name} schedule.")
    print(f"  betas shape: {betas.shape}, range: ({betas.min():.4f}, {betas.max():.4f})")
    print(f"  alphas_cumprod shape: {alphas_cumprod.shape}, range: ({alphas_cumprod.min():.4f}, {alphas_cumprod.max():.4f})")

    # --- Helper Function to Extract Values for a Batch ---
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """ Extracts values from 'a' at indices 't' and reshapes for broadcasting. """
        batch_size = t.shape[0]
        # Use t's device for indexing, ensure 'a' is on the same device or CPU
        # Gather using t.long() as indices
        out = a.to(t.device).gather(0, t.long())
        # Reshape to (batch_size, 1, 1, 1) for broadcasting with image shape
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # --- Forward Process (q - adding noise) ---
    # q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
    def q_sample(x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Samples x_t by adding noise to x_0 according to the schedule. """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # equation: sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # --- Display Info ---
    schedule_info = mo.md(f"""
    Diffusion constants calculated for **T = {timesteps}** steps using the **{_schedule_name}** schedule.  
     - `betas`: Shape `{betas.shape}`, Range `({betas.min():.4f}, {betas.max():.4f})`  
     - `alphas_cumprod`: Shape `{alphas_cumprod.shape}`, Range `({alphas_cumprod.min():.4f}, {alphas_cumprod.max():.4f})`  
    """)
    schedule_info
    return q_sample, timesteps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 💾 Training Setup (Model Instantiation, Optimizer, Checkpointing)""")
    return


@app.cell(hide_code=True)
def checkpoints(
    Dict,
    Optional,
    checkpoint_dir_text,
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
    """ Cell for managing model checkpoints: saving, loading, and plotting loss."""

    _chkpt_dir = checkpoint_dir_text.value
    _load_flag = load_checkpoint_checkbox.value

    # --- Marimo State for Persistent Training Info ---
    # Stores list of (epoch, avg_loss) tuples
    get_epoch_log, set_epoch_log = mo.state([])
    # Stores the epoch number to start training from (1 if new, N+1 if loaded)
    get_start_epoch, set_start_epoch = mo.state(1)

    def display_loss_plot() -> Optional[plt.Figure]:
        """ Renders a plot of the training loss stored in Marimo state. """
        epoch_log = get_epoch_log()
        if not epoch_log:
            return None # Don't display plot if no log exists
        try:
            epochs, losses = zip(*epoch_log)
            fig, ax = plt.subplots(figsize=(8, 3))
            # Dynamically adjust y-axis limit based on initial loss
            y_top_limit = losses[0] * 1.2 if _losses else 1.0
            plt.ylim(bottom=0, top=max(0.1, y_top_limit)) # Ensure non-negative limit
            ax.plot(epochs, losses, marker="o", linestyle="-", markersize=4, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss")
            ax.set_title("Training Loss Over Epochs")
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            # plt.show() # Not needed in Marimo, just return the figure
            return fig # Return the matplotlib figure object
        except Exception as e:
            print(f"Error plotting loss: {e}")
            return mo.md(f"Error plotting loss: {e}").callout(kind="danger")

    def save_checkpoint(
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        avg_epoch_loss: float,
        config: Dict # Store relevant config settings
    ) -> None:
        """ Saves model, optimizer state, epoch, loss, and config to a file. """
        os.makedirs(_chkpt_dir, exist_ok=True) # Ensure directory exists
        # Use a consistent filename structure
        base_filename = f"unet_T{config.get('T', 'unk')}_{config.get('schedule', 'unk')}"
        checkpoint_path = os.path.join(_chkpt_dir, f"{base_filename}_epoch_{epoch}.pth")
        try:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "epoch_log": get_epoch_log(), # Save the loss history
                    "config": config, # Save relevant configuration
                },
                checkpoint_path,
            )
            print(f"✅ Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")

    def _read_checkpoint_data(checkpoint: Dict, model: nn.Module, optimizer: optim.Optimizer) -> str:
        """ Helper to load state dicts and update Marimo state. """
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        prev_log = checkpoint.get("epoch_log", []) # Handle older checkpoints
        set_epoch_log(prev_log)
        prev_epoch = checkpoint["epoch"]
        set_start_epoch(prev_epoch + 1) # Resume from the next epoch

        # Option: uncomment to force LR to current UI setting after loading
        # current_lr = learning_rate_slider.value
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = current_lr

        config_info = checkpoint.get("config", {})
        msg = (f"👍 Checkpoint loaded from epoch {prev_epoch} "
                f"(Loss: {checkpoint['loss']:.4f}). "
                f"Config: T={config_info.get('T', 'N/A')}, "
                f"Schedule={config_info.get('schedule', 'N/A')}. "
                f"Resuming from epoch {get_start_epoch()}.")
        print(msg)
        return msg

    def load_latest_checkpoint(model: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> str:
        """ Finds and loads the latest valid checkpoint file from the directory. """
        status_msg = "▶️ Starting training from scratch."
        set_start_epoch(1) # Reset start epoch
        set_epoch_log([]) # Reset epoch log

        if not _load_flag:
            return "Load checkpoint disabled. Starting from scratch."

        if not os.path.isdir(_chkpt_dir):
            return f"⚠️ Checkpoint directory not found: `{_chkpt_dir}`. Starting from scratch."

        try:
            # Find files matching a pattern (e.g., *epoch_*.pth)
            files = [
                f for f in os.listdir(_chkpt_dir)
                if f.endswith(".pth") and "_epoch_" in f
            ]

            if not files:
                return f"ℹ️ No checkpoint files found in `{_chkpt_dir}`. Starting from scratch."

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
            if loaded_config.get('T') != timesteps_slider.value:
                mismatches.append(f"Timesteps (T): Loaded={loaded_config.get('T')} vs UI={timesteps_slider.value}")
            if loaded_config.get('schedule') != noise_schedule_dropdown.value:
                mismatches.append(f"Schedule: Loaded={loaded_config.get('schedule')} vs UI={noise_schedule_dropdown.value}")
            if mismatches:
                 warning = ("⚠️ Checkpoint loaded, but config mismatches detected:\n - " +
                             "\n - ".join(mismatches) +
                             "\n Training will continue, but results might be unexpected.")
                 print(warning)
                 status_msg = _read_checkpoint_data(checkpoint, model, optimizer) + "\n" + warning
            else:
                 status_msg = _read_checkpoint_data(checkpoint, model, optimizer)

        except FileNotFoundError:
             status_msg = "⚠️ Checkpoint file specified but not found. Starting from scratch."
        except Exception as e:
            status_msg = f"❌ Error loading checkpoint from `{files[0]}`: {e}. Training from scratch."
            set_start_epoch(1) # Reset state if loading failed
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
    """ Cell to define training setup: Instanciating model, optimizer, loss_fn, reading prev checkpoints """
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU. Training will be slow.")

    # --- Instantiate Model ---
    # Use configuration values passed from previous cells
    model = UNet(
        in_channels=channels,
        out_channels=channels,
        time_emb_dim=128,       # Can be adjusted, ensure consistency if loading checkpoints
        base_dim=32,            # Start with 32, maybe increase to 64 if using GPU and need more capacity
        dim_mults=(1, 2, 4),    # (1, 2) for faster/lighter, (1, 2, 4) for more capacity
        num_res_blocks=2,
        dropout=0.1
    ).to(device)

    _model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated with {_model_param_count:,} trainable parameters.")

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate_slider.value, weight_decay=1e-4)
    print(f"Optimizer: AdamW, LR={learning_rate_slider.value:.1e}")

    # --- Loss Function ---
    loss_fn = nn.MSELoss() # Standard loss for predicting noise
    print("Loss Function: MSELoss")

    # --- Load Checkpoint (if enabled and available) ---
    # Pass the current device to map the loaded tensors correctly
    _checkpoint_status_msg = load_latest_checkpoint(model, optimizer, device)

    # Display setup status in Marimo UI
    training_setup_info = mo.vstack([
        dataloader_info, # Show DataLoader status first
        mo.md("---"),
        mo.md(f"**Device:** `{device}`"),
        mo.md(f"**Model:** U-Net with {_model_param_count:,} parameters."),
        mo.md(f"**Optimizer:** AdamW (LR={learning_rate_slider.value:.1e})"),
        mo.md("**Loss Function:** MSELoss"),
        mo.md(f"**Checkpoint Status:** {_checkpoint_status_msg}"),
        mo.md("---"),
        mo.md("**Training Loss History:**"),
        display_loss_plot() # Display the loss plot
    ])
    training_setup_info
    return device, loss_fn, model, optimizer, training_setup_info


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🚀 Training Loop""")
    return


@app.cell(hide_code=True)
def display_ui_conf(mo, training_setup_info, ui_config_layout):
    # Display the main config UI again for easy access
    # And the status panel from the setup cell
    mo.vstack([
        ui_config_layout,
        mo.md("---"),
        training_setup_info
    ])
    return


@app.cell
def _(
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
    mo.stop(
        not start_training_button.value,
        mo.md("▶️ Press the **'Start / Continue Training'** button above to begin training.")
    )

    def train_one_epoch(epoch: int, n_epochs: int, steps_per_epoch: int, dataloader: DataLoader) -> float:
        """ Trains the model for one epoch. """
        model.train() # Set model to training mode
        epoch_loss = 0.0
        step_start_time = time.time()

        # Use Marimo's progress bar
        with mo.status.progress_bar(total=steps_per_epoch, title=f"Epoch {epoch}/{n_epochs}") as bar:
            for step, batch in enumerate(dataloader):
                if step >= steps_per_epoch: break # Limit steps per epoch

                optimizer.zero_grad()
                batch = batch.to(device) # Move data to the correct device
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
                if True: # (step + 1) % 5 == 0: # Update less frequently for performance
                    _avg_loss_so_far = epoch_loss / (step + 1)
                    _steps_per_sec = (step + 1) / (time.time() - step_start_time)
                    bar.update(
                       subtitle=f"Step {step + 1}/{steps_per_epoch} | Loss: {_current_loss:.4f} | Avg Loss: {_avg_loss_so_far:.4f} | Steps/sec: {_steps_per_sec:.2f}"
                    )

            # Ensure final update for the bar
            _avg_loss_so_far = epoch_loss / steps_per_epoch
            _steps_per_sec = steps_per_epoch / (time.time() - step_start_time)
            bar.update(
                subtitle=f"Step {steps_per_epoch}/{steps_per_epoch} | Avg Loss: {_avg_loss_so_far:.4f} | Steps/sec: {_steps_per_sec:.2f}"
            )

        return epoch_loss / steps_per_epoch # Return average loss for the epoch


    def train(n_epochs: int, current_start_epoch):
        """ Main Training Execution for many epoch. """
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
            last_epoch = get_start_epoch() -1
            if last_epoch >= current_start_epoch:
                 print("Attempting to save checkpoint before exiting...")
                 chkpt_config = {
                     "T": timesteps,
                     "schedule": noise_schedule_dropdown.value
                 }
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


if __name__ == "__main__":
    app.run()
