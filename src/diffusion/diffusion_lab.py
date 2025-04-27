import marimo

app = marimo.App()

@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    import torch.nn.functional as TF # Need F for padding
    #from torchvision.utils import make_grid

    import numpy as np

    from PIL import Image, ImageDraw, ImageFilter
    import math
    import os
    import time
    import random

    from io import BytesIO

    import matplotlib.pyplot as plt

    # Ensure reproducibility (optional)
    # torch.manual_seed(42)
    # np.random.seed(42)
    # random.seed(42)
    return (
        BytesIO,
        DataLoader,
        Dataset,
        Image,
        ImageDraw,
        ImageFilter,
        T,
        TF,
        math,
        mo,
        nn,
        np,
        optim,
        os,
        plt,
        random,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Image Generation using Diffusion Models

        This notebook demonstrates the basics of training a diffusion model for image generation.
        We will generate synthetic 128x128 images of Platonic solids on a plane on-the-fly.

        **Note:** Training diffusion models is computationally expensive. CPU training will be **very slow**.
        This notebook is primarily for educational purposes to understand the components involved.
        Expect long training times for visible results.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Configuration""")
    return


@app.cell(hide_code=True)
def _(mo):
    # --- General Settings ---
    image_size_slider = mo.ui.slider( 128, 128, value=128, label="Image Size (Fixed)")  # Fixed for this demo

    # Reduce batch size for CPU memory
    batch_size_slider = mo.ui.slider(2, 32, step=2, value=4, label="Batch Size")

    # Limited epochs for demo purposes on CPU
    n_epochs_slider = mo.ui.slider(1, 100, step=1, value=10, label="Number of Epochs")
    # Limited epochs for demo purposes on CPU
    n_step_slider = mo.ui.slider(1, 512, step=1, value=256, label="Number of Steps per Epochs")

    learning_rate_slider = mo.ui.number(1e-4, 1e-2, step=1e-5, value=5e-4, label="Learning Rate")


    # --- Diffusion Settings ---
    timesteps_slider = mo.ui.slider(50, 1000, step=50, value=200, label="Diffusion Timesteps (T)")  # Fewer steps for faster CPU training/sampling


    # --- Data Generation Settings ---
    color_mode = mo.ui.dropdown(["Grayscale", "RGB"], value="RGB", label="Image Mode")
    num_solids_range_slider = mo.ui.range_slider(
        1, 8, value=(2, 5), label="Number of Solids per Image")


    # --- Model & Checkpointing ---

    # model_choice = mo.ui.dropdown(["UNet_Simple", "UNet_Standard"], value="UNet_Simple", label="Model Architecture") # Placeholder if adding more models

    checkpoint_dir = mo.ui.text("models\\diffusion", label="Checkpoint Directory")
    save_freq_epochs_slider = mo.ui.slider(1, 10, value=1, label="Save Checkpoint Every (Epochs)")
    load_checkpoint_flag_checkbox = mo.ui.checkbox(value=True, label="Load Latest Checkpoint if Available")

    # --- UI Controls ---
    start_training_button = mo.ui.run_button(label="Start/Continue Training")

    generate_samples_button = mo.ui.run_button(label="Generate Samples")

    sampler_choice = mo.ui.dropdown( ["DDPM", "DDIM"], value="DDIM", label="Sampler Type")

    num_samples_to_generate = mo.ui.slider(1, 16, value=4, label="Number of Samples to Generate")

    ddim_eta = mo.ui.slider(0.0, 1.0, step=0.1, value=0.0, label="DDIM Eta (0=Deterministic)")

    # --- Display Configured Values ---
    mo.vstack([
            mo.md("### UI Configuration"),
            mo.hstack([
                mo.vstack([
                    mo.md("#### General & Diffusion"),
                    image_size_slider,
                    batch_size_slider,
                    n_epochs_slider,
                    n_step_slider,
                    learning_rate_slider,
                    timesteps_slider,
                    mo.md("  "),
                ]),
                mo.vstack([
                    mo.md("#### Data Generation"),
                    color_mode,
                    num_solids_range_slider,
                    mo.md("  "),
                    mo.md("#### Model & Checkpointing"),
                    # model_choice,
                    checkpoint_dir,
                    save_freq_epochs_slider,
                    load_checkpoint_flag_checkbox,
                    mo.md("  "),
                ]),
            ]),
            mo.md("#### Sampling Controls"),
            sampler_choice,
            ddim_eta,
            num_samples_to_generate,
            mo.md("---"),
            mo.hstack([start_training_button, generate_samples_button], justify="start")
        ])
    return (
        batch_size_slider,
        checkpoint_dir,
        color_mode,
        ddim_eta,
        generate_samples_button,
        image_size_slider,
        learning_rate_slider,
        load_checkpoint_flag_checkbox,
        n_epochs_slider,
        n_step_slider,
        num_solids_range_slider,
        sampler_choice,
        save_freq_epochs_slider,
        start_training_button,
        timesteps_slider,
    )


@app.cell(hide_code=True)
def _(
    batch_size_slider,
    checkpoint_dir,
    color_mode,
    ddim_eta,
    image_size_slider,
    learning_rate_slider,
    load_checkpoint_flag_checkbox,
    mo,
    n_epochs_slider,
    n_step_slider,
    num_solids_range_slider,
    sampler_choice,
    save_freq_epochs_slider,
    timesteps_slider,
):
    mo.md(f"""
    ### Resume
    **Current Configuration:**  
    - Image Size: {image_size_slider.value}x{image_size_slider.value}  
    - Batch Size: {batch_size_slider.value}  
    - Epochs x Steps: {n_epochs_slider.value}x{n_step_slider.value}  
    - Learning Rate: {learning_rate_slider.value}  
    - Timesteps (T): {timesteps_slider.value}  
    - Color Mode: {color_mode.value} ({("Color" if color_mode.value == "RGB" else "Grayscale")})  
    - Solids per Image: {num_solids_range_slider.value[0]} to {num_solids_range_slider.value[1]}  
    - Checkpoint Dir: `{checkpoint_dir.value}`  
    - Save Freq: {save_freq_epochs_slider.value} epochs  
    - Load Checkpoint: {load_checkpoint_flag_checkbox.value}  
    - Sampler: {sampler_choice.value.upper()} (Eta: {ddim_eta.value if sampler_choice.value == "ddim" else "N/A"})  
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Synthetic Data Generation""")
    return


@app.cell(hide_code=True)
def _(BytesIO, Image, ImageDraw, ImageFilter, math, mo, random):
    # --- Platonic Solid Definitions (Simplified 2D Projections) ---

    # Coordinates are roughly centered around (0,0) with scale ~0.8
    # These are illustrative and not geometrically perfect projections.

    PLATONIC_SOLIDS = {
        "Tetrahedron": [
            (0.0, -0.8),
            (0.7, 0.4),
            (-0.7, 0.4),  # Triangle base
            # Maybe add internal lines later if needed
        ],
        "Cube": [
            (-0.5, -0.5),
            (0.5, -0.5),
            (0.5, 0.5),
            (-0.5, 0.5),  # Front face
            # Could add lines for perspective, e.g., (-0.3, -0.7), (0.7, -0.3), (0.7, 0.7)
        ],
        "Octahedron": [
            (0.0, -0.8),
            (0.7, 0.0),
            (0.0, 0.8),
            (-0.7, 0.0),  # Diamond shape
        ],
        "Dodecahedron": [  # Simplified Pentagon
            (0.0, -0.8),
            (0.76, -0.25),
            (0.47, 0.65),
            (-0.47, 0.65),
            (-0.76, -0.25),
        ],
        "Icosahedron": [  # Simplified Hexagon (proxy)
            (0.4, -0.7),
            (0.8, 0.0),
            (0.4, 0.7),
            (-0.4, 0.7),
            (-0.8, 0.0),
            (-0.4, -0.7),
        ],
    }


    SOLID_NAMES = list(PLATONIC_SOLIDS.keys())


    # Assign vibrant colors (can be customized)

    # Using simple RGB tuples

    VIBRANT_COLORS = {
        "Tetrahedron": (255, 0, 0),  # Red
        "Cube": (0, 0, 255),  # Blue
        "Octahedron": (0, 255, 0),  # Green
        "Dodecahedron": (255, 255, 0),  # Yellow
        "Icosahedron": (255, 0, 255),  # Magenta
    }

    GRAY_COLORS = { name: (random.randint(80, 200),) * 3 for name in SOLID_NAMES }  # Use RGB tuple even for gray


    def get_solid_vertices(solid_name, center_x, center_y, size):
        vertices = PLATONIC_SOLIDS[solid_name]
        scaled_vertices = [(center_x + x * size, center_y + y * size) for x, y in vertices ]
        return scaled_vertices


    # --- Image Generation Function ---
    def generate_scene_image(img_size=128, mode="RGB", num_solids_min=2, num_solids_max=5):
        """Generates an image with random Platonic solids on a plane."""
        if mode == "Grayscale":
            background_color = (50,)  # Dark gray plane
            final_mode = "Grayscale"

        else:  # RGB
            background_color = (40, 40, 60)  # Dark blueish plane
            final_mode = "RGB"

        image = Image.new(mode, (img_size, img_size), background_color)
        draw = ImageDraw.Draw(image)
        num_solids = random.randint(num_solids_min, num_solids_max)
        placed_solids = []

        for _ in range(num_solids):
            solid_name = random.choice(SOLID_NAMES)
            max_size = img_size * 0.25  # Max solid size relative to image
            min_size = img_size * 0.1
            solid_size = random.uniform(min_size, max_size)

            # Try to place without too much overlap (simple check)
            placed = False

            for _ in range(10):  # Max placement attempts
                center_x = random.uniform(solid_size, img_size - solid_size)
                center_y = random.uniform(solid_size, img_size - solid_size)

                # Basic overlap check (bounding box)
                too_close = False
                for _, px, py, psize in placed_solids:
                    dist = math.sqrt((center_x - px) ** 2 + (center_y - py) ** 2)
                    if dist < (solid_size + psize) * 0.7:  # Allow some overlap
                        too_close = True
                        break

                if not too_close:
                    placed = True
                    break

            if not placed:
                continue  # Skip if can't place

            vertices = get_solid_vertices( solid_name, center_x, center_y, solid_size )

            # --- Color ---
            if mode == "Grayscale":
                solid_color = GRAY_COLORS[solid_name][0]  # Single gray value
            else:
                solid_color = VIBRANT_COLORS[solid_name]

            # --- Shadow ---
            shadow_offset_x = solid_size * 0.15
            shadow_offset_y = solid_size * 0.15

            shadow_vertices = [(x + shadow_offset_x, y + shadow_offset_y) for x, y in vertices ]

            shadow_color = (
                tuple(max(0, c - 40) for c in background_color)
                if mode == "RGB"
                else max(0, background_color[0] - 40)
            )

            # Draw shadow first (as a separate blurred image blended)
            shadow_img = Image.new(
                mode, (img_size, img_size), (0,) * len(background_color)
            )  # Transparent shadow layer? No, use bg

            shadow_draw = ImageDraw.Draw(shadow_img)

            shadow_draw.polygon(shadow_vertices, fill=shadow_color)

            shadow_img = shadow_img.filter(
                ImageFilter.GaussianBlur(radius=solid_size * 0.05)
            )

            # Blend shadow onto main image (use background where shadow is 0)
            # Create mask from shadow alpha (or value if L) is tricky with PIL blend.
            # Simpler: draw shadow directly, then object over it.
            draw.polygon(shadow_vertices, fill=shadow_color)

            # --- Solid ---

            # Simple lighting: slightly lighter fill, maybe outline
            outline_color = (
                tuple(min(255, c + 30) for c in solid_color)
                if mode == "RGB"
                else min(255, solid_color + 30)
            )

            draw.polygon(vertices, fill=solid_color, outline=outline_color)
            placed_solids.append((solid_name, center_x, center_y, solid_size))

        # Convert to final mode if needed (e.g. if intermediary was RGBA)

        if image.mode != final_mode:
            image = image.convert(final_mode)

        return image


    # --- Test Generation ---


    def display_samples(
        num=4, img_size=128, mode="RGB", num_solids_min=2, num_solids_max=5
    ):
        images = [
            generate_scene_image(img_size, mode, num_solids_min, num_solids_max)
            for _ in range(num)
        ]

        # Convert PIL images to format Marimo can display easily (e.g., BytesIO PNG)
        byte_streams = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            byte_streams.append(buf.getvalue())

        # Use mo.hstack for layout
        return mo.hstack([mo.image(bs) for bs in byte_streams])


    return display_samples, generate_scene_image


@app.cell
def _(generate_samples_button):
    generate_samples_button
    return


@app.cell(hide_code=True)
def _(
    color_mode,
    display_samples,
    generate_samples_button,
    image_size_slider,
    mo,
    num_solids_range_slider,
):
    # if the button hasn't been clicked, don't run.
    mo.stop(not generate_samples_button.value, mo.md("Press 'Start/Continue Training' button to run 🔥"))

    # Display sample images based on current config
    display_samples(
        num=4,
        img_size=image_size_slider.value,
        mode=color_mode.value,
        num_solids_min=num_solids_range_slider.value[0],
        num_solids_max=num_solids_range_slider.value[1]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 3. Dataset and DataLoader""")
    return


@app.cell(hide_code=True)
def _(
    DataLoader,
    Dataset,
    T,
    batch_size_slider,
    color_mode,
    generate_scene_image,
    image_size_slider,
    mo,
    n_step_slider,
    num_solids_range_slider,
    torch,
):
    # Normalization transform: map image pixels from [0, 255] to [-1, 1]
    # Adjust number of channels based on color mode

    channels = 3 if color_mode.value == "RGB" else 1

    transform = T.Compose(
        [
            T.ToTensor(),  # Converts PIL image (H, W, C) or (H, W) to (C, H, W) tensor & scales to [0, 1]
            T.Lambda(lambda t: (t * 2) - 1),  # Scale from [0, 1] to [-1, 1]
        ]
    )


    class SyntheticSolidsDataset(Dataset):
        def __init__(
            self,
            img_size=128,
            mode="RGB",
            count=1000,
            transform=None,
            num_solids_min=2,
            num_solids_max=5,
        ):
            self.img_size = img_size
            self.mode = mode
            self.count = count  # Virtual dataset size
            self.transform = transform
            self.num_solids_min = num_solids_min
            self.num_solids_max = num_solids_max

        def __len__(self):
            # Return a large number, images are generated on the fly
            return self.count

        def __getitem__(self, idx):
            img = generate_scene_image(
                self.img_size, self.mode, self.num_solids_min, self.num_solids_max
            )

            if self.transform:
                img = self.transform(img)

            # Ensure correct channel dimension even for grayscale

            if self.mode == "Grayscale" and img.dim() == 2:
                img = img.unsqueeze(0)  # Add channel dim: (H, W) -> (1, H, W)

            # Sanity check for channel numbers

            expected_channels = 1 if self.mode == "Grayscale" else 3

            if img.shape[0] != expected_channels:
                # This indicates an issue in generation or transform
                # Fallback: Create a dummy tensor of correct shape
                print(f"Warning: Image channel mismatch. Expected {expected_channels}, got {img.shape[0]}. Returning dummy.")

                return torch.zeros(expected_channels, self.img_size, self.img_size)

            return img


    # Instantiate dataset - set count high enough for epochs * steps_per_epoch
    # The count here doesn't preload data, just defines iterable length.
    # For demonstration, maybe 1000 is enough if batches are small.
    # Adjust based on batch_size_slider and n_epochs_slider?
    # Effective dataset size per epoch = len(dataloader) * batch_size_slider
    # Let's make it reasonably large for training.

    #virtual_dataset_size = 1024
    virtual_dataset_size = n_step_slider.value * batch_size_slider.value

    dataset = SyntheticSolidsDataset(
        img_size=image_size_slider.value,
        mode=color_mode.value,
        count=virtual_dataset_size,
        transform=transform,
        num_solids_min=num_solids_range_slider.value[0],
        num_solids_max=num_solids_range_slider.value[1],
    )


    # Create DataLoader
    # IMPORTANT for Windows: num_workers MUST be 0 if data generation is complex
    # or uses non-thread-safe libraries within the __getitem__

    def get_dataloader():
        return DataLoader(
            dataset,
            batch_size=batch_size_slider.value,
            shuffle=True,
            num_workers=0,  # Crucial for Windows + on-the-fly generation
            drop_last=True,  # Drop last batch if smaller than batch_size
        )


    # Example: Get one batch to check shapes
    try:
        temp_loader = get_dataloader(4)  # Use a small fixed batch size for testing
        sample_batch = next(iter(temp_loader))
        batch_shape = sample_batch.shape
        data_loader_status = f"DataLoader ready. Sample batch shape: {batch_shape}"
        del temp_loader  # Clean up

    except Exception as e:
        batch_shape = None
        data_loader_status = f"Error creating DataLoader or getting batch: {e}"
        print(data_loader_status)

    dataloader_info = mo.md(f"**DataLoader Status:** {data_loader_status}")
    mo.vstack([
            mo.md("Dataset and DataLoader created."),
            dataloader_info
        ])
    return channels, get_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Diffusion Model (U-Net)""")
    return


@app.cell(hide_code=True)
def _(TF, math, mo, nn, torch):
    # --- Time Embedding ---
    class SinusoidalPosEmb(nn.Module):
        # (Keep this class exactly as it was)
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            device = x.device
            half_dim = self.dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            emb = x[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            return emb

    # --- Building Block: Residual Block with Time Embedding ---
    class ResidualBlock(nn.Module):
        # (Keep this class exactly as it was)
        def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.norm1 = nn.GroupNorm(8, out_channels)
            self.act1 = nn.SiLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.norm2 = nn.GroupNorm(8, out_channels)
            self.act2 = nn.SiLU()
            self.dropout = nn.Dropout(dropout)

            if in_channels != out_channels:
                self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.skip_connection = nn.Identity()

            if time_emb_dim is not None:
                self.time_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_emb_dim, out_channels)
                )
            else:
                self.time_mlp = None

        def forward(self, x, time_emb=None):
            h = self.act1(self.norm1(self.conv1(x)))
            if self.time_mlp is not None and time_emb is not None:
                time_encoding = self.time_mlp(time_emb)
                h = h + time_encoding.unsqueeze(-1).unsqueeze(-1)
            h = self.dropout(self.act2(self.norm2(self.conv2(h))))
            return h + self.skip_connection(x)

    # --- U-Net Architecture (Corrected Version) ---
    class UNet(nn.Module):
        def __init__(
            self,
            in_channels=3,
            out_channels=3,
            time_emb_dim=128,
            base_dim=32,
            dim_mults=(1, 2), # Defaulting to smaller for CPU
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.base_dim = base_dim
            self.dim_mults = dim_mults

            # 1. Time Embedding Projection
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.SiLU(),
                nn.Linear(time_emb_dim * 4, time_emb_dim),
            )

            # 2. Initial Convolution
            self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

            # 3. Encoder Path (Downsampling)
            self.downs = nn.ModuleList([])
            num_resolutions = len(dim_mults)
            current_dim = base_dim
            print(f"--- UNet Init: Encoder ---")
            for i in range(num_resolutions):
                dim_out = base_dim * dim_mults[i]
                print(f"  Level {i}: {current_dim} -> {dim_out} channels")
                self.downs.append(
                    nn.ModuleList(
                        [
                            ResidualBlock(current_dim, dim_out, time_emb_dim),
                            ResidualBlock(dim_out, dim_out, time_emb_dim),
                            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2, padding=1)
                            if i < num_resolutions - 1 else nn.Identity(), # Downsample except last level
                        ]
                    )
                )
                current_dim = dim_out # Update current_dim for the next level

            # 4. Bottleneck
            print(f"--- UNet Init: Bottleneck ({current_dim} channels) ---")
            self.mid_block1 = ResidualBlock(current_dim, current_dim, time_emb_dim)
            self.mid_block2 = ResidualBlock(current_dim, current_dim, time_emb_dim)
            bottleneck_dim = current_dim # Save bottleneck dimension for decoder start

            # 5. Decoder Path (Upsampling)
            self.ups = nn.ModuleList([])
            decoder_input_dim = bottleneck_dim # Start with bottleneck output dimension
            print(f"--- UNet Init: Decoder ---")
            for i in reversed(range(num_resolutions)): # e.g., i = 1, then i = 0 for mults=(1,2)
                # Dimension of the skip connection coming from the corresponding encoder level 'i'
                # The skip connection is the output of the *second* ResBlock at encoder level 'i'
                dim_encoder_out = base_dim * dim_mults[i] # i=1 -> 64; i=0 -> 32
                dim_skip = dim_encoder_out

                # Target output dimension for *this* decoder level
                dim_target_out = base_dim * dim_mults[i - 1] if i > 0 else base_dim # i=1 -> 32; i=0 -> 32 (base_dim)

                print(f"  Level {i}: Input {decoder_input_dim}, Skip {dim_skip}, Target Out {dim_target_out}")

                # Upsampling layer: Takes decoder_input_dim, outputs dim_target_out channels
                # Only add ConvTranspose if not the last decoder stage (highest resolution)
                is_first_decoder_level = (i == num_resolutions - 1) # Check if it's the bottleneck connection
                upsample_layer = nn.ConvTranspose2d(decoder_input_dim, dim_target_out, kernel_size=4, stride=2, padding=1) # Upsamples x2

                # Calculate input channels for the first ResBlock after concatenation
                res1_input_ch = dim_target_out + dim_skip # Channels after upsample + skip channels

                self.ups.append(
                    nn.ModuleList(
                        [
                            upsample_layer,
                            ResidualBlock(res1_input_ch, dim_target_out, time_emb_dim), # Input: Upsampled + Skip, Output: target_out
                            ResidualBlock(dim_target_out, dim_target_out, time_emb_dim), # Input: target_out, Output: target_out
                        ]
                    )
                )

                # The input dimension for the NEXT decoder level's upsample layer is the output of this level
                decoder_input_dim = dim_target_out

            # 6. Final Convolution Layer
            print(f"--- UNet Init: Final Conv ({base_dim} -> {out_channels} channels) ---")
            self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)
            print(f"--- UNet Init: Complete ---")


        def forward(self, x, time):
            # 1. Get time embedding
            t_emb = self.time_mlp(time)

            # 2. Initial convolution
            x = self.init_conv(x) # (B, base_dim, H, W)
            skips = [x] # Store for skip connections (after init_conv)

            # 3. Encoder path
            # print("--- Encoder Forward ---")
            for i, (res1, res2, downsample) in enumerate(self.downs):
                # print(f" Encoder Level {i} Input: {x.shape}")
                x = res1(x, t_emb)
                # print(f"  Res1 Out: {x.shape}")
                x = res2(x, t_emb)
                # print(f"  Res2 Out: {x.shape}")
                skips.append(x) # Store output *before* downsampling for skip connection
                x = downsample(x)
                # print(f"  Downsample Out: {x.shape}")

            # 4. Bottleneck
            # print(f"--- Bottleneck Forward ({x.shape}) ---")
            x = self.mid_block1(x, t_emb)
            # print(f"  Mid1 Out: {x.shape}")
            x = self.mid_block2(x, t_emb)
            # print(f"  Mid2 Out: {x.shape}")

            # 5. Decoder path
            # print("--- Decoder Forward ---")
            current_level_input = x # Start with bottleneck output
            for i, (upsample, res1, res2) in enumerate(self.ups):
                # print(f" Decoder Level {i} Input: {current_level_input.shape}")

                # Apply Upsampling (ConvTranspose2D)
                current_level_input = upsample(current_level_input)
                # print(f"  Upsample Out: {current_level_input.shape}")

                # Get skip connection from corresponding encoder level
                skip_connection = skips.pop()
                # print(f"  Skip Connection: {skip_connection.shape}")

                # Pad current_level_input if spatial dimensions don't match skip_connection
                # (Can happen with certain kernel/stride/padding combos, esp. odd dimensions)
                diffY = skip_connection.size()[2] - current_level_input.size()[2]
                diffX = skip_connection.size()[3] - current_level_input.size()[3]
                if diffX != 0 or diffY != 0:
                    current_level_input = TF.pad(current_level_input,
                                                [diffX // 2, diffX - diffX // 2,
                                                 diffY // 2, diffY - diffY // 2])
                    # print(f"  Padded Upsampled: {current_level_input.shape}")

                # Concatenate upsampled tensor and skip connection
                h = torch.cat((current_level_input, skip_connection), dim=1)
                # print(f"  Concatenated (h): {h.shape} (Expected input for Res1: {res1.conv1.in_channels})") # Debug print

                # Apply Residual Blocks
                h = res1(h, t_emb)
                # print(f"  Res1 Out: {h.shape}")
                current_level_input = res2(h, t_emb) # Output of this level becomes input for the next
                # print(f"  Res2 Out: {current_level_input.shape}")

            # 6. Final convolution
            # print(f"--- Final Conv Input ({current_level_input.shape}) ---")
            out = self.final_conv(current_level_input)
            # print(f"--- Final Output ({out.shape}) ---")
            return out


    # === Code to Instantiate and Test (Keep this part in the cell) ===
    try:
        # Determine in/out channels from config (ensure these variables exist in the cell's scope)
        # You might need to get them from the ui_elements dictionary if running interactively
        # _in_channels = 1 if ui_elements['color_mode'].value == "L" else 3
        _in_channels = 3 # Assuming RGB for now, adjust if needed
        _out_channels = _in_channels
        _image_size_slider_val = 128 # Assuming 128, adjust if needed
        _timesteps_slider_val = 200   # Assuming 200, adjust if needed

        print("\nInstantiating Model...")
        # Simple model for CPU
        _model = UNet(
            in_channels=_in_channels,
            out_channels=_out_channels,
            time_emb_dim=128,
            base_dim=32,
            dim_mults=(1, 2), # Smaller U-Net
        )

        print("\nTesting Forward Pass...")
        # Test forward pass with dummy data
        _dummy_x = torch.randn(2, _in_channels, _image_size_slider_val, _image_size_slider_val)
        _dummy_t = torch.randint(0, _timesteps_slider_val, (2,)).long()
        _output = _model(_dummy_x, _dummy_t)
        model_status = f"U-Net Instantiated. Output shape: {_output.shape}"
        print(f"\n{model_status}")

        del _model, _dummy_x, _dummy_t, _output  # Clean up

    except Exception as e:
        model_status = f"Error instantiating or testing U-Net: {e}"
        print(f"\nERROR: {model_status}")
        # Print traceback for detailed error location
        import traceback
        traceback.print_exc()


    # Display status in Marimo
    model_info = mo.md(f"**Model Status:** {model_status}")
    mo.vstack([
            mo.md("U-Net model defined."),
            model_info
        ])

    # Make necessary modules available outside the cell if needed elsewhere
    # return SinusoidalPosEmb, ResidualBlock, UNet, model_info, model_status, torch
    return (UNet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Diffusion Process Utilities""")
    return


@app.cell(hide_code=True)
def _(mo, np, timesteps_slider, torch):
    # --- Diffusion Scheduler ---

    def linear_beta_schedule(ts):
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, ts)


    # Pre-calculate diffusion constants
    timesteps_slider_value = timesteps_slider.value
    betas = linear_beta_schedule(timesteps_slider_value)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0]), alphas_cumprod[:-1]]
    )  # F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)


    # Helper function to extract specific values for a batch of timesteps_slider t
    def extract(a, t, x_shape):
        batch_size_slider = t.shape[0]
        out = a.gather(-1, t.cpu())  # Gather based on t indices
        return out.reshape(batch_size_slider, *((1,) * (len(x_shape) - 1))).to(
            t.device
        )  # Reshape to broadcast

    # --- Forward Process (q - adding noise) ---
    # q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
    def q_sample(x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(
            torch.sqrt(alphas_cumprod), t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            torch.sqrt(1.0 - alphas_cumprod), t, x_start.shape
        )
        # equation: sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )


    # --- Reverse Process (p - sampling) ---
    # DDPM Sampling Step: p(x_{t-1} | x_t)
    # Uses model prediction epsilon_theta(x_t, t)
    @torch.no_grad()
    def p_sample_ddpm(model, x_t, t, t_index):
        betas_t = extract(betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            torch.sqrt(1.0 - alphas_cumprod), t, x_t.shape
        )
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / alphas), t, x_t.shape)

        # Equation 11 from DDPM paper:
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_cumprod_t) * epsilon_theta(x_t, t)) + sigma_t * z
        # where z is noise N(0, I), and sigma_t^2 = beta_t or (1-alpha_cumprod_{t-1})/(1-alpha_cumprod_t) * beta_t

        # Use model to predict noise
        predicted_noise = model(x_t, t)

        # Calculate mean of p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean  # No noise added at the last step
        else:
            # Calculate variance (sigma_t^2 * I) - use fixed variance beta_t
            posterior_variance_t = extract(
                betas, t, x_t.shape
            )  # sigma_t^2 = beta_t
            noise = torch.randn_like(x_t)

            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    # DDPM Sampling Loop
    @torch.no_grad()
    def p_sample_loop_ddpm(model, shape, device):
        b = shape[0]  # Batch size
        # Start from pure noise x_T ~ N(0, I)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in reversed(range(0, timesteps_slider_value)):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)
            img = p_sample_ddpm(model, img, t_tensor, i)
            # Optionally store intermediate steps
            # if i % 50 == 0: imgs.append(img.cpu())
        imgs.append(img.cpu())  # Store final result
        return imgs


    # DDIM Sampling Step
    @torch.no_grad()
    def p_sample_ddim(model, x_t, t, t_prev, eta=0.0):
        # Predict noise and x_0
        predicted_noise = model(x_t, t)
        alpha_cumprod_t = extract(alphas_cumprod, t, x_t.shape)
        alpha_cumprod_t_prev = extract(alphas_cumprod, t_prev, x_t.shape)

        # Equation (12) from DDIM paper: predicted x_0
        pred_x0 = (
            x_t - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise
        ) / torch.sqrt(alpha_cumprod_t)

        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)  # Clip predicted x0

        # Equation (12) cont.: direction pointing to x_t
        pred_dir_xt = (
            torch.sqrt(1.0 - alpha_cumprod_t_prev - (eta**2)) * predicted_noise
        )

        # Equation (12) cont.: random noise term
        sigma_t = eta * torch.sqrt(
            (1.0 - alpha_cumprod_t_prev)
            / (1.0 - alpha_cumprod_t)
            * (1.0 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

        noise = torch.randn_like(x_t) if torch.any(t > 0) else 0  # No noise at t=0

        # Combine: x_{t-1} = sqrt(alpha_cumprod_{t-1}) * pred_x0 + pred_dir_xt + sigma_t * noise
        x_prev = (
            torch.sqrt(alpha_cumprod_t_prev) * pred_x0
            + pred_dir_xt
            + sigma_t * noise
        )
        return x_prev


    # DDIM Sampling Loop
    @torch.no_grad()
    def p_sample_loop_ddim(model, shape, device, num_inference_steps=50, eta=0.0):
        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []
        # Define DDIM timesteps_slider (subset of T)
        step_ratio = timesteps_slider_value // num_inference_steps
        ddim_timesteps_slider = (np.arange(0, num_inference_steps) * step_ratio).round()
        ddim_timesteps_slider = ddim_timesteps_slider.astype(int) + 1  # Start from 1
        ddim_timesteps_slider = np.flip(ddim_timesteps_slider)  # Reverse: T -> 1

        # Ensure unique timesteps_slider and add step 0 if needed.
        # This needs refinement based on DDIM paper's sequence (c.f. Eq 12): t_i = (i * T / S)
        # Let's use linspace for better sequence generation T -> 0
        ddim_timesteps_slider_seq = np.linspace(
            0, timesteps_slider_value - 1, num_inference_steps
        ).astype(int)

        ddim_timesteps_slider_seq = np.flip(ddim_timesteps_slider_seq)  # T-1, ..., 0

        times = torch.from_numpy(ddim_timesteps_slider_seq.copy()).long().to(device)

        times_prev = (
            torch.from_numpy(np.concatenate([[0], ddim_timesteps_slider_seq[:-1]]).copy())
            .long()
            .to(device)
        )  # t_{i-1}

        for i, (t, t_prev) in enumerate(zip(times, times_prev)):
            time_tensor = torch.full((b,), t, device=device, dtype=torch.long)

            time_prev_tensor = torch.full(
                (b,), t_prev, device=device, dtype=torch.long
            )  # Needed if extract uses tensor directly

            img = p_sample_ddim(model, img, time_tensor, time_prev_tensor, eta=eta)

            # Optionally store intermediate steps
            # if i % (num_inference_steps // 5) == 0: imgs.append(img.cpu())
        imgs.append(img.cpu())  # Store final result
        return imgs

    mo.md(f"""
    Diffusion constants calculated for timesteps={timesteps_slider_value}.  
    Beta schedule: Linear from {betas.min():.1e} to {betas.max():.2f}.
    """)
    return q_sample, timesteps_slider_value


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 6. Training Setup""")
    return


@app.cell(hide_code=True)
def _(
    UNet,
    channels,
    checkpoint_dir,
    learning_rate_slider,
    load_checkpoint_flag_checkbox,
    mo,
    n_epochs_slider,
    nn,
    optim,
    os,
    plt,
    start_training_button,
    torch,
):
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
        #plt.show()
        return plt.gcf()

    # Use mo.state to manage the training execution and outputs persistently
    # Initialize state variables if they don't exist
    get_epoch_log, set_epoch_log = mo.state([])  # Store tuples of (epoch, avg_loss)
    get_start_epoch, set_start_epoch = mo.state(1)  # Store start_epoch

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = mo.md(f"**Using Device:** `{device}`")

    # --- Instantiate Model ---
    # Ensure channels match current config
    _in_channels = channels  # From dataset cell
    _out_channels = _in_channels

    model = UNet(
        in_channels=_in_channels,
        out_channels=_out_channels,
        time_emb_dim=128,  # Can be adjusted
        base_dim=32,  # Smaller for CPU: 32 or even 16
        dim_mults=(1, 2),  # Fewer levels: (1, 2) or (1, 2, 4)
    ).to(device)

    model_param_count = sum(p.numel() for p in model.parameters())
    model_instance_info = mo.md(f"Model instance created with {model_param_count:,} parameters.")


    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate_slider.value, weight_decay=1e-4)

    # --- Loss Function ---
    loss_fn = nn.MSELoss()

    # --- Checkpoint Handling ---
    chkpt_dir = checkpoint_dir.value
    os.makedirs(chkpt_dir, exist_ok=True)
    latest_checkpoint = None

    checkpoint_status = mo.md("No checkpoint found.").callout(kind="warn")


    # Find latest checkpoint file
    try:
        checkpoint_files = [f for f in os.listdir(chkpt_dir) if f.endswith(".pth")]

        if checkpoint_files:
            checkpoint_files.sort(
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )  # Sort by epoch number

            latest_checkpoint = os.path.join(chkpt_dir, checkpoint_files[-1])

    except OSError as e:
        print(f"Could not access checkpoint directory {chkpt_dir}: {e}")

    if load_checkpoint_flag_checkbox.value and latest_checkpoint:
        try:
            print(f"Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            set_epoch_log(checkpoint["epoch_log"])
            prev_epoch = checkpoint["epoch"]
            set_start_epoch(prev_epoch +1)  # Start from next epoch
            # Ensure LR matches current setting if desired, or use saved LR
            # optimizer.param_groups[0]['lr'] = learning_rate_slider.value

            print(f"Checkpoint loaded. Resuming from epoch {prev_epoch}.")
            checkpoint_status = mo.md(f"Loaded checkpoint from epoch {prev_epoch} (Loss: {checkpoint["loss"]:.6f}).")

        except Exception as e:
            checkpoint_status = mo.md(f"Error loading checkpoint: {e}").callout(kind="danger")
    elif load_checkpoint_flag_checkbox.value and not latest_checkpoint:
        checkpoint_status = mo.md("Load requested, but no checkpoint found.").callout(kind="warn")
    else:
        checkpoint_status = mo.md("Starting training from scratch.").callout(kind="info")


    def save_model(epoch, avg_epoch_loss):
        checkpoint_path = os.path.join(chkpt_dir, f"diffusion_model_epoch_{epoch}.pth")
        try:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    # Save epoch log for plotting continuity
                    "epoch_log": get_epoch_log(),
                    # You might want to save other things like config, learning rate etc.
                },
                checkpoint_path,
            )

            print(f"Checkpoint saved to {checkpoint_path}")

        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    # Display setup status
    setup_status = mo.vstack(
        [
            device_info,
            model_instance_info,
            mo.md(f"Optimizer: AdamW, LR: {learning_rate_slider.value}"),
            mo.md(f"Loss Function: MSELoss"),
            checkpoint_status
        ]
    )
    mo.vstack(
        [
            setup_status,
            (mo.vstack([display_epoch_log(get_epoch_log()),f"Last Loss: {get_epoch_log()[-1][1]:.4f}"]) if get_epoch_log() else "No epoch data"),
            (mo.hstack([start_training_button,
            f"From epoch {get_start_epoch()} to {n_epochs_slider.value}"
            ]) if get_start_epoch() <= n_epochs_slider.value else f"Model is trainned after {get_start_epoch() -1 } epochs. (change epoch number slider to continue: {n_epochs_slider.value}).")
        ]
    )
    return (
        device,
        get_epoch_log,
        get_start_epoch,
        loss_fn,
        model,
        optimizer,
        save_model,
        set_epoch_log,
        set_start_epoch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 7. Training Loop""")
    return


@app.cell(hide_code=True)
def _(
    device,
    get_dataloader,
    get_epoch_log,
    get_start_epoch,
    loss_fn,
    mo,
    model,
    n_epochs_slider,
    optimizer,
    q_sample,
    save_freq_epochs_slider,
    save_model,
    set_epoch_log,
    set_start_epoch,
    start_training_button,
    time,
    timesteps_slider_value,
    torch,
):
    # if the button hasn't been clicked, don't run.
    mo.stop(not start_training_button.value, mo.md(("Press 'Start/Continue Training' button to run 🔥" if get_start_epoch() <= n_epochs_slider.value else "Model is trained 🟢")))


    def progress(bar, epoch, step, steps_per_epoch, current_loss, epoch_loss):
        bar.update(
            title=f"Epoch {epoch}/{n_epochs_slider.value} Step {step + 1}/{steps_per_epoch}",
            subtitle=f"Step Batch Loss: {current_loss:.4f}, Epoch Avg Loss: {epoch_loss / (step + 1):.4f}",
        )


    # --- Prepare DataLoader ---
    # Must be created here to use the current batch_size_slider value
    dataloader = get_dataloader()
    steps_per_epoch = len(dataloader)

    # --- Training Loop ---
    model.train()
    total_start_time = time.time()
    for epoch in range(get_start_epoch(), n_epochs_slider.value + 1):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        with mo.status.progress_bar(title="Steps", total=steps_per_epoch) as bar:
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                batch = batch.to(device)
                b_size = batch.shape[0]

                # 1. Sample random timesteps_slider t for each image in the batch
                t = torch.randint(
                    0, timesteps_slider_value, (b_size,), device=device
                ).long()

                # 2. Sample noise eps ~ N(0, I)
                noise = torch.randn_like(batch)

                # 3. Calculate x_t = q_sample(x_0, t, eps) (noisy image)
                x_noisy = q_sample(x_start=batch, t=t, noise=noise)

                # 4. Predict noise using the model: eps_theta = model(x_t, t)
                predicted_noise = model(x_noisy, t)

                # 5. Calculate loss: MSE(eps, eps_theta)
                loss = loss_fn(noise, predicted_noise)

                # 6. Backpropagate and update optimizer
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Update progress
                current_loss = loss.item()
                progress(
                    bar, epoch, step, steps_per_epoch, current_loss, epoch_loss
                )

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / steps_per_epoch
        epoch_duration = time.time() - epoch_start_time

        # Log epoch loss
        set_epoch_log(get_epoch_log() + [(epoch, avg_epoch_loss)])

        # Update plot
        print(
            f"Epoch {epoch}/{n_epochs_slider.value}, Avg Loss: {avg_epoch_loss:.4f},  Time: {epoch_duration:.2f}s"
        )
        set_start_epoch(epoch+1)  # allow restart by increasing n_epochs_slider
        # --- Checkpointing ---
        if (
            (epoch +1) % save_freq_epochs_slider.value == 0
            or epoch == n_epochs_slider.value
        ):
            save_model(epoch, avg_epoch_loss)

    # --- Training Finished ---
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    final_status = f"Training finished after {n_epochs_slider.value} epochs ({total_duration:.2f}s)."
    print(final_status)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 8. end""")
    return


if __name__ == "__main__":
    app.run()
