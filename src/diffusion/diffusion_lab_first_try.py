# diffusion_lab.py
import marimo


__generated_with = "0.3.8"

app = marimo.App(width="medium")



@app.cell

def __():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from torch.utils.data import Dataset, DataLoader

    import torchvision.transforms as T

    from torchvision.utils import make_grid


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

        make_grid,
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



@app.cell

def __(mo):
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



@app.cell

def __(mo):

    mo.md("## 1. Configuration")
    return



@app.cell

def __(mo):

    # --- General Settings ---

    image_size = mo.ui.slider(128, 128, value=128, label="Image Size (Fixed)", disabled=True) # Fixed for this demo

    # Reduce batch size for CPU memory

    batch_size = mo.ui.slider(2, 32, step=2, value=4, label="Batch Size")

    # Limited epochs for demo purposes on CPU

    n_epochs = mo.ui.slider(1, 100, step=1, value=10, label="Number of Epochs")

    learning_rate = mo.ui.number(1e-4, 1e-2, step=1e-5, value=2e-4, label="Learning Rate")


    # --- Diffusion Settings ---

    timesteps = mo.ui.slider(50, 1000, step=50, value=200, label="Diffusion Timesteps (T)") # Fewer steps for faster CPU training/sampling


    # --- Data Generation Settings ---

    color_mode = mo.ui.dropdown({"Grayscale": "L", "Color": "RGB"}, value="RGB", label="Image Mode")

    num_solids_range = mo.ui.range_slider(1, 8, value=(2, 5), label="Number of Solids per Image")


    # --- Model & Checkpointing ---

    # model_choice = mo.ui.dropdown(["UNet_Simple", "UNet_Standard"], value="UNet_Simple", label="Model Architecture") # Placeholder if adding more models

    checkpoint_dir = mo.ui.text("diffusion_checkpoints", label="Checkpoint Directory")

    save_freq_epochs = mo.ui.slider(1, 10, value=2, label="Save Checkpoint Every (Epochs)")

    load_checkpoint_flag = mo.ui.checkbox(value=False, label="Load Latest Checkpoint if Available")


    # --- UI Controls ---

    start_training_button = mo.ui.button(label="Start/Continue Training")

    generate_samples_button = mo.ui.button(label="Generate Samples")

    sampler_choice = mo.ui.dropdown({"DDPM": "ddpm", "DDIM": "ddim"}, value="ddim", label="Sampler Type")

    num_samples_to_generate = mo.ui.slider(1, 16, value=4, label="Number of Samples to Generate")

    ddim_eta = mo.ui.slider(0.0, 1.0, step=0.1, value=0.0, label="DDIM Eta (0=Deterministic)")


    # --- Display Configured Values ---

    config_display = mo.md(

        f"""

        **Current Configuration:**

        - Image Size: {image_size.value}x{image_size.value}

        - Batch Size: {batch_size.value}

        - Epochs: {n_epochs.value}

        - Learning Rate: {learning_rate.value}

        - Timesteps (T): {timesteps.value}

        - Image Mode: {color_mode.value} ({('Color' if color_mode.value == 'RGB' else 'Grayscale')})

        - Solids per Image: {num_solids_range.value[0]} to {num_solids_range.value[1]}

        - Checkpoint Dir: `{checkpoint_dir.value}`

        - Save Freq: {save_freq_epochs.value} epochs

        - Load Checkpoint: {load_checkpoint_flag.value}

        - Sampler: {sampler_choice.value.upper()} (Eta: {ddim_eta.value if sampler_choice.value == 'ddim' else 'N/A'})

        """
    )


    # Group UI elements

    config_ui = mo.vstack([

        mo.md("### General & Diffusion"),

        image_size, batch_size, n_epochs, learning_rate, timesteps,

        mo.md("### Data Generation"),

        color_mode, num_solids_range,

        mo.md("### Model & Checkpointing"),

        # model_choice,

        checkpoint_dir, save_freq_epochs, load_checkpoint_flag,

        mo.md("### Sampling Controls"),

        sampler_choice, ddim_eta, num_samples_to_generate,

        mo.md("---"),

        config_display

    ])


    control_buttons = mo.hstack([start_training_button, generate_samples_button], justify='start')


    ui_elements = {

        "image_size": image_size, "batch_size": batch_size, "n_epochs": n_epochs,

        "learning_rate": learning_rate, "timesteps": timesteps, "color_mode": color_mode,

        "num_solids_range": num_solids_range, "checkpoint_dir": checkpoint_dir,

        "save_freq_epochs": save_freq_epochs, "load_checkpoint_flag": load_checkpoint_flag,

        "start_training_button": start_training_button, "generate_samples_button": generate_samples_button,

        "sampler_choice": sampler_choice, "num_samples_to_generate": num_samples_to_generate,

        "ddim_eta": ddim_eta

    }
    return (

        batch_size,

        checkpoint_dir,
        color_mode,

        config_display,

        config_ui,

        control_buttons,
        ddim_eta,

        generate_samples_button,

        image_size,

        learning_rate,

        load_checkpoint_flag,
        n_epochs,

        num_samples_to_generate,

        num_solids_range,
        sampler_choice,

        save_freq_epochs,

        start_training_button,
        timesteps,
        ui_elements,
    )



@app.cell

def __(config_ui, control_buttons, mo):

    # Display the UI elements

    mo.vstack([

        config_ui,

        control_buttons

    ])
    return



@app.cell

def __(mo):

    mo.md("## 2. Synthetic Data Generation")
    return



@app.cell

def __(ImageFilter, Image, ImageDraw, math, np, random):

    # --- Platonic Solid Definitions (Simplified 2D Projections) ---

    # Coordinates are roughly centered around (0,0) with scale ~0.8

    # These are illustrative and not geometrically perfect projections.


    PLATONIC_SOLIDS = {

        "Tetrahedron": [

            (0.0, -0.8), (0.7, 0.4), (-0.7, 0.4) # Triangle base

            # Maybe add internal lines later if needed

        ],

        "Cube": [

            (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5) # Front face

            # Could add lines for perspective, e.g., (-0.3, -0.7), (0.7, -0.3), (0.7, 0.7)

        ],

        "Octahedron": [

             (0.0, -0.8), (0.7, 0.0), (0.0, 0.8), (-0.7, 0.0) # Diamond shape

        ],

        "Dodecahedron": [ # Simplified Pentagon

            (0.0, -0.8), (0.76, -0.25), (0.47, 0.65), (-0.47, 0.65), (-0.76, -0.25)

        ],

        "Icosahedron": [ # Simplified Hexagon (proxy)

            (0.4, -0.7), (0.8, 0.0), (0.4, 0.7), (-0.4, 0.7), (-0.8, 0.0), (-0.4, -0.7)

        ]

    }


    SOLID_NAMES = list(PLATONIC_SOLIDS.keys())


    # Assign vibrant colors (can be customized)

    # Using simple RGB tuples

    VIBRANT_COLORS = {

        "Tetrahedron": (255, 0, 0),    # Red

        "Cube":        (0, 0, 255),    # Blue

        "Octahedron":  (0, 255, 0),    # Green

        "Dodecahedron": (255, 255, 0), # Yellow

        "Icosahedron": (255, 0, 255), # Magenta

    }

    GRAY_COLORS = { name: (random.randint(80, 200),) * 3 for name in SOLID_NAMES } # Use RGB tuple even for gray



    def get_solid_vertices(solid_name, center_x, center_y, size):

        vertices = PLATONIC_SOLIDS[solid_name]

        scaled_vertices = [

            (center_x + x * size, center_y + y * size) for x, y in vertices

        ]

        return scaled_vertices


    # --- Image Generation Function ---

    def generate_scene_image(img_size=128, mode="RGB", num_solids_min=2, num_solids_max=5):

        """Generates an image with random Platonic solids on a plane."""

        if mode == "L":

            background_color = (50,) # Dark gray plane

            final_mode = "L"

        else: # RGB

            background_color = (40, 40, 60) # Dark blueish plane

            final_mode = "RGB"


        image = Image.new(mode, (img_size, img_size), background_color)

        draw = ImageDraw.Draw(image)


        num_solids = random.randint(num_solids_min, num_solids_max)

        placed_solids = []


        for _ in range(num_solids):

            solid_name = random.choice(SOLID_NAMES)

            max_size = img_size * 0.25 # Max solid size relative to image

            min_size = img_size * 0.1

            solid_size = random.uniform(min_size, max_size)


            # Try to place without too much overlap (simple check)

            placed = False

            for _ in range(10): # Max placement attempts

                center_x = random.uniform(solid_size, img_size - solid_size)

                center_y = random.uniform(solid_size, img_size - solid_size)


                # Basic overlap check (bounding box)

                too_close = False

                for _, px, py, psize in placed_solids:

                    dist = math.sqrt((center_x - px)**2 + (center_y - py)**2)

                    if dist < (solid_size + psize) * 0.7: # Allow some overlap

                         too_close = True

                         break

                if not too_close:

                    placed = True

                    break


            if not placed: continue # Skip if can't place


            vertices = get_solid_vertices(solid_name, center_x, center_y, solid_size)


            # --- Color ---

            if mode == "L":

                solid_color = GRAY_COLORS[solid_name][0] # Single gray value

            else:

                solid_color = VIBRANT_COLORS[solid_name]


            # --- Shadow ---

            shadow_offset_x = solid_size * 0.15

            shadow_offset_y = solid_size * 0.15

            shadow_vertices = [(x + shadow_offset_x, y + shadow_offset_y) for x, y in vertices]

            shadow_color = tuple(max(0, c - 40) for c in background_color) if mode == "RGB" else max(0, background_color[0] - 40)


            # Draw shadow first (as a separate blurred image blended)

            shadow_img = Image.new(mode, (img_size, img_size), (0,) * len(background_color)) # Transparent shadow layer? No, use bg

            shadow_draw = ImageDraw.Draw(shadow_img)

            shadow_draw.polygon(shadow_vertices, fill=shadow_color)

            shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=solid_size * 0.05))

            # Blend shadow onto main image (use background where shadow is 0)

            # Create mask from shadow alpha (or value if L) is tricky with PIL blend.

            # Simpler: draw shadow directly, then object over it.

            draw.polygon(shadow_vertices, fill=shadow_color)



            # --- Solid ---

            # Simple lighting: slightly lighter fill, maybe outline

            outline_color = tuple(min(255, c + 30) for c in solid_color) if mode == "RGB" else min(255, solid_color + 30)

            draw.polygon(vertices, fill=solid_color, outline=outline_color)



            placed_solids.append((solid_name, center_x, center_y, solid_size))


        # Convert to final mode if needed (e.g. if intermediary was RGBA)

        if image.mode != final_mode:

            image = image.convert(final_mode)


        return image


    # --- Test Generation ---

    def display_samples(num=4, img_size=128, mode="RGB", num_solids_min=2, num_solids_max=5):

        images = [generate_scene_image(img_size, mode, num_solids_min, num_solids_max) for _ in range(num)]

        # Convert PIL images to format Marimo can display easily (e.g., BytesIO PNG)

        byte_streams = []

        for img in images:

            buf = BytesIO()

            img.save(buf, format='PNG')

            buf.seek(0)

            byte_streams.append(buf.getvalue())


        # Use mo.hstack for layout

        return mo.hstack([mo.image(bs) for bs in byte_streams])
    return (

        GRAY_COLORS,

        PLATONIC_SOLIDS,

        SOLID_NAMES,

        VIBRANT_COLORS,

        display_samples,

        generate_scene_image,

        get_solid_vertices,
    )



@app.cell
def __(

    BytesIO,
    color_mode,

    display_samples,

    image_size,
    mo,

    num_solids_range,

):

    mo.md("### Generated Data Samples")

    # Display sample images based on current config

    _sample_display = display_samples(

        num=4,

        img_size=image_size.value,

        mode=color_mode.value,

        num_solids_min=num_solids_range.value[0],

        num_solids_max=num_solids_range.value[1]
    )

    _sample_display
    return



@app.cell

def __(mo):

    mo.md("## 3. Dataset and DataLoader")
    return



@app.cell
def __(

    Dataset,

    T,
    color_mode,

    generate_scene_image,

    image_size,
    np,

    num_solids_range,
    torch,

):

    # Normalization transform: map image pixels from [0, 255] to [-1, 1]

    # Adjust number of channels based on color mode

    channels = 3 if color_mode.value == "RGB" else 1

    transform = T.Compose([

        T.ToTensor(),  # Converts PIL image (H, W, C) or (H, W) to (C, H, W) tensor & scales to [0, 1]

        T.Lambda(lambda t: (t * 2) - 1) # Scale from [0, 1] to [-1, 1]

    ])


    class SyntheticSolidsDataset(Dataset):

        def __init__(self, img_size=128, mode="RGB", count=1000, transform=None, num_solids_min=2, num_solids_max=5):

            self.img_size = img_size

            self.mode = mode

            self.count = count # Virtual dataset size

            self.transform = transform

            self.num_solids_min = num_solids_min

            self.num_solids_max = num_solids_max


        def __len__(self):

            # Return a large number, images are generated on the fly
            return self.count


        def __getitem__(self, idx):

            img = generate_scene_image(

                self.img_size,
                self.mode,
                self.num_solids_min,

                self.num_solids_max
            )

            if self.transform:

                img = self.transform(img)

            # Ensure correct channel dimension even for grayscale

            if self.mode == 'L' and img.dim() == 2:

                img = img.unsqueeze(0) # Add channel dim: (H, W) -> (1, H, W)


            # Sanity check for channel numbers

            expected_channels = 1 if self.mode == 'L' else 3

            if img.shape[0] != expected_channels:

                 # This indicates an issue in generation or transform

                 # Fallback: Create a dummy tensor of correct shape

                 print(f"Warning: Image channel mismatch. Expected {expected_channels}, got {img.shape[0]}. Returning dummy.")

                 return torch.zeros(expected_channels, self.img_size, self.img_size)


            return img


    # Instantiate dataset - set count high enough for epochs * steps_per_epoch

    # The count here doesn't preload data, just defines iterable length.

    # For demonstration, maybe 1000 is enough if batches are small.

    # Adjust based on batch_size and n_epochs?

    # Effective dataset size per epoch = len(dataloader) * batch_size

    # Let's make it reasonably large for training.

    virtual_dataset_size = 1024


    dataset = SyntheticSolidsDataset(

        img_size=image_size.value,

        mode=color_mode.value,

        count=virtual_dataset_size,

        transform=transform,

        num_solids_min=num_solids_range.value[0],

        num_solids_max=num_solids_range.value[1]
    )


    # Create DataLoader

    # IMPORTANT for Windows: num_workers MUST be 0 if data generation is complex

    # or uses non-thread-safe libraries within the __getitem__

    def get_dataloader(current_batch_size):

        return DataLoader(
            dataset,

            batch_size=current_batch_size,

            shuffle=True,

            num_workers=0, # Crucial for Windows + on-the-fly generation

            drop_last=True # Drop last batch if smaller than batch_size
        )


    # Example: Get one batch to check shapes

    try:

        temp_loader = get_dataloader(4) # Use a small fixed batch size for testing

        sample_batch = next(iter(temp_loader))

        batch_shape = sample_batch.shape

        data_loader_status = f"DataLoader ready. Sample batch shape: {batch_shape}"

        del temp_loader # Clean up

    except Exception as e:

        batch_shape = None

        data_loader_status = f"Error creating DataLoader or getting batch: {e}"
        print(data_loader_status)



    dataloader_info = mo.md(f"**DataLoader Status:** {data_loader_status}")
    return (

        DataLoader,

        SyntheticSolidsDataset,

        batch_shape,
        channels,
        data_loader_status,
        dataloader_info,
        dataset,

        get_dataloader,
        transform,

        virtual_dataset_size,
    )



@app.cell

def __(dataloader_info, mo):

    mo.vstack([

        mo.md("Dataset and DataLoader created."),
        dataloader_info

    ])
    return



@app.cell

def __(mo):

    mo.md("## 4. Diffusion Model (U-Net)")
    return



@app.cell

def __(math, nn, torch):

    # --- Time Embedding ---

    class SinusoidalPosEmb(nn.Module):

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

        def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.1):
            super().__init__()

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

            self.norm1 = nn.GroupNorm(8, out_channels) # GroupNorm often better for diffusion

            self.act1 = nn.SiLU() # Swish/SiLU is common


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

                # Add time embedding (broadcast across spatial dimensions)

                h = h + time_encoding.unsqueeze(-1).unsqueeze(-1) # Or use .view()


            h = self.dropout(self.act2(self.norm2(self.conv2(h))))


            return h + self.skip_connection(x)


    # --- U-Net Architecture ---

    # Simplified U-Net suitable for 128x128 and CPU demo

    # Fewer channels, fewer blocks than SOTA models

    class UNet(nn.Module):

        def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128, base_dim=32, dim_mults=(1, 2, 4)):
            super().__init__()


            self.time_mlp = nn.Sequential(

                SinusoidalPosEmb(time_emb_dim),

                nn.Linear(time_emb_dim, time_emb_dim * 4),

                nn.SiLU(),

                nn.Linear(time_emb_dim * 4, time_emb_dim)
            )


            dims = [base_dim] + [base_dim * m for m in dim_mults]

            in_out_dims = list(zip(dims[:-1], dims[1:])) # [(32, 64), (64, 128)] if dim_mults=(1,2,4) -> dims=[32,32,64,128] -> in_out=[(32,32),(32,64),(64,128)] - Correct this logic.


            # Correct dimension calculation

            current_dim = base_dim

            self.dims = [current_dim]

            for m in dim_mults:

                self.dims.append(current_dim * m)

            in_out_dims = list(zip(self.dims[:-1], self.dims[1:])) # e.g., [(32, 64), (64, 128), (128, 256)] if base=32, mults=(2,4,8)

                                                                    # If base=32, mults=(1,2,4) -> dims=[32, 32, 64, 128] -> in_out=[(32, 32), (32, 64), (64, 128)]


            self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)


            self.downs = nn.ModuleList([])

            self.ups = nn.ModuleList([])

            num_resolutions = len(dim_mults)


            # --- Encoder ---

            for i, (dim_in, dim_out) in enumerate(in_out_dims):

                is_last = (i == (num_resolutions - 1))

                self.downs.append(nn.ModuleList([

                    ResidualBlock(dim_in, dim_out, time_emb_dim),

                    ResidualBlock(dim_out, dim_out, time_emb_dim),

                    nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2, padding=1) if not is_last else nn.Identity() # Downsample

                ]))


            # --- Bottleneck ---

            mid_dim = self.dims[-1]

            self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim)

            # Add Attention maybe? For simplicity, skip for now.

            self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_emb_dim)


            # --- Decoder ---

            for i, (dim_out, dim_in) in enumerate(reversed(in_out_dims)): # Reverse dims for upsampling

                is_last = (i == (num_resolutions - 1))

                self.ups.append(nn.ModuleList([

                     # Input channels: dim_in from previous up + dim_out from skip connection

                    ResidualBlock(dim_in + dim_out, dim_in, time_emb_dim),

                    ResidualBlock(dim_in, dim_in, time_emb_dim),

                     # Upsample (scale_factor=2) then Conv, or ConvTranspose

                    nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1) if not is_last else nn.Identity()

                ]))



            # Final Layer

            self.final_conv = nn.Sequential(

                nn.GroupNorm(8, base_dim),

                nn.SiLU(),

                nn.Conv2d(base_dim, out_channels, kernel_size=1) # Predict noise
            )



        def forward(self, x, time):

            # x: (B, C, H, W)

            # time: (B,) tensor of timesteps


            t_emb = self.time_mlp(time) # (B, time_emb_dim)


            x = self.init_conv(x) # (B, base_dim, H, W)

            h = [x] # Store activations for skip connections


            # --- Encoder ---

            for i, (res1, res2, downsample) in enumerate(self.downs):

                x = res1(x, t_emb)

                h.append(x)

                x = res2(x, t_emb)

                h.append(x)

                x = downsample(x)



            # --- Bottleneck ---

            x = self.mid_block1(x, t_emb)

            x = self.mid_block2(x, t_emb)


            # --- Decoder ---

            for i, (res1, res2, upsample) in enumerate(self.ups):

                # Pop skip connections in reverse order (most recent first)

                skip1 = h.pop()

                skip2 = h.pop()

                # Concatenate skip connection (make sure channels match)

                # We stored two activations per level in downs, need to match the input to res1

                # Input to res1 should be dim_in + dim_out.

                # x (from previous upsample or bottleneck) has dim_in channels.

                # skip connection (skip2 here?) should have dim_out channels.

                x = torch.cat((x, skip2), dim=1) # Check channel dims carefully

                x = res1(x, t_emb)


                # Maybe only need one skip connection per level? Let's adjust.

                # Store only after res2 before downsampling in encoder.

                # Redo forward pass logic and skip connection storage.


            # --- Corrected Forward Pass ---

            t_emb = self.time_mlp(time)

            x = self.init_conv(x) # (B, base_dim, H, W)

            skips = [x]


            # Encoder

            for i, (res1, res2, downsample) in enumerate(self.downs):

                 x = res1(x, t_emb)

                 x = res2(x, t_emb)

                 skips.append(x) # Store before downsampling

                 x = downsample(x)


            # Bottleneck

            x = self.mid_block1(x, t_emb)

            x = self.mid_block2(x, t_emb)


            # Decoder

            for i, (res1, res2, upsample) in enumerate(self.ups):

                skip_connection = skips.pop()

                x = torch.cat((x, skip_connection), dim=1) # Concatenate along channel dim

                x = res1(x, t_emb)

                x = res2(x, t_emb)

                x = upsample(x)


            # Final layer requires input dim = base_dim

            # After last upsample block, x should have base_dim channels

            # The last 'upsample' in the loop might be Identity(), check logic.

            # Let's refine the upsampling loop structure. Need to handle channel dimensions carefully.


            # --- Revised U-Net Logic (Simpler Structure Example) ---

            # Let's use a more standard U-Net block structure for clarity.


            # Re-define based on common patterns.

            del self.downs, self.ups, self.mid_block1, self.mid_block2, self.final_conv

            self.downs = nn.ModuleList([])

            self.ups = nn.ModuleList([])

            num_resolutions = len(dim_mults)


            # --- Encoder ---

            current_dim = base_dim

            for i in range(num_resolutions):

                dim_out = base_dim * dim_mults[i]

                self.downs.append(nn.ModuleList([

                    ResidualBlock(current_dim, dim_out, time_emb_dim),

                    ResidualBlock(dim_out, dim_out, time_emb_dim),

                    nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2, padding=1) if i < num_resolutions - 1 else nn.Identity()

                ]))

                current_dim = dim_out


            # --- Bottleneck ---

            self.mid_block1 = ResidualBlock(current_dim, current_dim, time_emb_dim)

            self.mid_block2 = ResidualBlock(current_dim, current_dim, time_emb_dim)


            # --- Decoder ---

            for i in reversed(range(num_resolutions)):

                dim_in = base_dim * dim_mults[i]

                dim_out = base_dim * dim_mults[i-1] if i > 0 else base_dim

                dim_skip = base_dim * dim_mults[i] # Dim from skip connection


                self.ups.append(nn.ModuleList([

                    # Upsample first or last? ConvTranspose handles upsampling.

                    nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1) if i > 0 else nn.Identity(),

                    ResidualBlock(dim_out + dim_skip, dim_out, time_emb_dim), # Input: Upsampled + Skip

                    ResidualBlock(dim_out, dim_out, time_emb_dim)

                ]))

                current_dim = dim_out # This is wrong. current_dim is needed for next iter



            # Final Layer

            self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)


            # --- Forward Pass (Revised Structure) ---

            def forward(self, x, time):

                t_emb = self.time_mlp(time)

                x = self.init_conv(x)

                skips = []


                # Encoder

                current_level_input = x

                for i, (res1, res2, downsample) in enumerate(self.downs):

                    h = res1(current_level_input, t_emb)

                    h = res2(h, t_emb)

                    skips.append(h) # Store output before downsampling

                    current_level_input = downsample(h)


                # Bottleneck

                x = self.mid_block1(current_level_input, t_emb)

                x = self.mid_block2(x, t_emb)


                # Decoder

                current_level_input = x

                for i, (upsample, res1, res2) in enumerate(self.ups):

                    current_level_input = upsample(current_level_input)

                    skip_connection = skips.pop()

                    # Ensure spatial dims match after upsample/skip before concat

                    diffY = skip_connection.size()[2] - current_level_input.size()[2]

                    diffX = skip_connection.size()[3] - current_level_input.size()[3]

                    # Pad current_level_input if needed (common U-Net pattern)

                    current_level_input = nn.functional.pad(current_level_input,

                                                            [diffX // 2, diffX - diffX // 2,

                                                             diffY // 2, diffY - diffY // 2])


                    h = torch.cat((current_level_input, skip_connection), dim=1)

                    h = res1(h, t_emb)

                    current_level_input = res2(h, t_emb) # Output of this level


                # Final Conv

                return self.final_conv(current_level_input)



    # Instantiate model to check structure

    try:

        # Determine in/out channels from config

        _in_channels = 1 if color_mode.value == "L" else 3

        _out_channels = _in_channels


        # Simple model for CPU

        _model = UNet(in_channels=_in_channels, out_channels=_out_channels, base_dim=32, dim_mults=(1, 2)) # Smaller U-Net


        # Test forward pass with dummy data

        _dummy_x = torch.randn(2, _in_channels, image_size.value, image_size.value)

        _dummy_t = torch.randint(0, timesteps.value, (2,)).long()

        _output = _model(_dummy_x, _dummy_t)

        model_status = f"U-Net Instantiated. Output shape: {_output.shape}"

        del _model, _dummy_x, _dummy_t, _output # Clean up

    except Exception as e:

        model_status = f"Error instantiating or testing U-Net: {e}"
        print(model_status)


    model_info = mo.md(f"**Model Status:** {model_status}")
    return (

        ResidualBlock,

        SinusoidalPosEmb,

        UNet,
        model_info,
        model_status,
        torch,
    )



@app.cell

def __(model_info, mo):

    mo.vstack([

        mo.md("U-Net model defined."),
        model_info

    ])
    return



@app.cell

def __(mo):

    mo.md("## 5. Diffusion Process Utilities")
    return



@app.cell

def __(math, timesteps, torch):

    # --- Diffusion Scheduler ---

    def linear_beta_schedule(T):

        beta_start = 1e-4

        beta_end = 0.02

        return torch.linspace(beta_start, beta_end, T)


    # Pre-calculate diffusion constants
    timesteps_value = timesteps.value

    betas = linear_beta_schedule(timesteps_value)

    alphas = 1. - betas

    alphas_cumprod = torch.cumprod(alphas, axis=0)

    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]]) # F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)


    # Helper function to extract specific values for a batch of timesteps t

    def extract(a, t, x_shape):

        batch_size = t.shape[0]

        out = a.gather(-1, t.cpu()) # Gather based on t indices

        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device) # Reshape to broadcast


    # --- Forward Process (q - adding noise) ---

    # q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)

    def q_sample(x_start, t, noise=None):

        if noise is None:

            noise = torch.randn_like(x_start)


        sqrt_alphas_cumprod_t = extract(torch.sqrt(alphas_cumprod), t, x_start.shape)

        sqrt_one_minus_alphas_cumprod_t = extract(

            torch.sqrt(1. - alphas_cumprod), t, x_start.shape
        )


        # equation: sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    # --- Reverse Process (p - sampling) ---


    # DDPM Sampling Step: p(x_{t-1} | x_t)

    # Uses model prediction epsilon_theta(x_t, t)

    @torch.no_grad()

    def p_sample_ddpm(model, x_t, t, t_index):

        betas_t = extract(betas, t, x_t.shape)

        sqrt_one_minus_alphas_cumprod_t = extract(

            torch.sqrt(1. - alphas_cumprod), t, x_t.shape
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

            return model_mean # No noise added at the last step

        else:

            # Calculate variance (sigma_t^2 * I) - use fixed variance beta_t

            posterior_variance_t = extract(betas, t, x_t.shape) # sigma_t^2 = beta_t

            noise = torch.randn_like(x_t)

            # Algorithm 2 line 4:

            return model_mean + torch.sqrt(posterior_variance_t) * noise


    # DDPM Sampling Loop

    @torch.no_grad()

    def p_sample_loop_ddpm(model, shape, device):

        b = shape[0] # Batch size

        # Start from pure noise x_T ~ N(0, I)

        img = torch.randn(shape, device=device)

        imgs = []


        for i in reversed(range(0, timesteps_value)):

            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)

            img = p_sample_ddpm(model, img, t_tensor, i)

            # Optionally store intermediate steps

            # if i % 50 == 0: imgs.append(img.cpu())

        imgs.append(img.cpu()) # Store final result

        return imgs



    # DDIM Sampling Step

    @torch.no_grad()

    def p_sample_ddim(model, x_t, t, t_prev, eta=0.0):

        # Predict noise and x_0

        predicted_noise = model(x_t, t)

        alpha_cumprod_t = extract(alphas_cumprod, t, x_t.shape)

        alpha_cumprod_t_prev = extract(alphas_cumprod, t_prev, x_t.shape)


        # Equation (12) from DDIM paper: predicted x_0

        pred_x0 = (x_t - torch.sqrt(1. - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

        pred_x0 = torch.clamp(pred_x0, -1., 1.) # Clip predicted x0


        # Equation (12) cont.: direction pointing to x_t

        pred_dir_xt = torch.sqrt(1. - alpha_cumprod_t_prev - (eta**2)) * predicted_noise


        # Equation (12) cont.: random noise term

        sigma_t = eta * torch.sqrt((1. - alpha_cumprod_t_prev) / (1. - alpha_cumprod_t) * (1. - alpha_cumprod_t / alpha_cumprod_t_prev))

        noise = torch.randn_like(x_t) if torch.any(t > 0) else 0 # No noise at t=0


        # Combine: x_{t-1} = sqrt(alpha_cumprod_{t-1}) * pred_x0 + pred_dir_xt + sigma_t * noise

        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigma_t * noise


        return x_prev


    # DDIM Sampling Loop

    @torch.no_grad()

    def p_sample_loop_ddim(model, shape, device, num_inference_steps=50, eta=0.0):

        b = shape[0]

        img = torch.randn(shape, device=device)

        imgs = []


        # Define DDIM timesteps (subset of T)

        step_ratio = timesteps_value // num_inference_steps

        ddim_timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()

        ddim_timesteps = ddim_timesteps.astype(int) + 1 # Start from 1

        ddim_timesteps = np.flip(ddim_timesteps) # Reverse: T -> 1


        # Ensure unique timesteps and add step 0 if needed.

        # This needs refinement based on DDIM paper's sequence (c.f. Eq 12): t_i = (i * T / S)

        # Let's use linspace for better sequence generation T -> 0

        ddim_timesteps_seq = np.linspace(0, timesteps_value - 1, num_inference_steps).astype(int)

        ddim_timesteps_seq = np.flip(ddim_timesteps_seq) # T-1, ..., 0


        times = torch.from_numpy(ddim_timesteps_seq.copy()).long().to(device)

        times_prev = torch.from_numpy(np.concatenate([[0], ddim_timesteps_seq[:-1]]).copy()).long().to(device) # t_{i-1}


        for i, (t, t_prev) in enumerate(zip(times, times_prev)):

            time_tensor = torch.full((b,), t, device=device, dtype=torch.long)

            time_prev_tensor = torch.full((b,), t_prev, device=device, dtype=torch.long) # Needed if extract uses tensor directly


            img = p_sample_ddim(model, img, time_tensor, time_prev_tensor, eta=eta)

            # Optionally store intermediate steps

            # if i % (num_inference_steps // 5) == 0: imgs.append(img.cpu())


        imgs.append(img.cpu()) # Store final result

        return imgs
    return (
        timesteps_value,
        alphas,
        alphas_cumprod,
        alphas_cumprod_prev,
        betas,
        ddim_timesteps_seq,
        extract,
        linear_beta_schedule,
        p_sample_ddim,
        p_sample_ddpm,
        p_sample_loop_ddim,
        p_sample_loop_ddpm,
        q_sample,
        torch,
    )


@app.cell
def __(T, betas, mo):

    mo.md(f"Diffusion constants calculated for T={T}. Beta schedule: Linear from {betas.min():.1e} to {betas.max():.2f}.")
    return



@app.cell

def __(mo):

    mo.md("## 6. Training Setup")
    return



@app.cell
def __(

    UNet,

    batch_size,
    channels,

    checkpoint_dir,
    color_mode,

    image_size,

    learning_rate,

    load_checkpoint_flag,
    mo,
    nn,
    optim,
    os,
    timesteps,
    torch,

):

    # --- Device Setup ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_info = mo.md(f"**Using Device:** `{device}`")


    # --- Instantiate Model ---

    # Ensure channels match current config

    _in_channels = channels # From dataset cell

    _out_channels = _in_channels

    model = UNet(

        in_channels=_in_channels,

        out_channels=_out_channels,

        time_emb_dim=128, # Can be adjusted

        base_dim=32,      # Smaller for CPU: 32 or even 16

        dim_mults=(1, 2)  # Fewer levels: (1, 2) or (1, 2, 4)

    ).to(device)

    model_param_count = sum(p.numel() for p in model.parameters())

    model_instance_info = mo.md(f"Model instance created with {model_param_count:,} parameters.")


    # --- Optimizer ---

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate.value, weight_decay=1e-4)


    # --- Loss Function ---

    loss_fn = nn.MSELoss()


    # --- Checkpoint Handling ---

    chkpt_dir = checkpoint_dir.value

    os.makedirs(chkpt_dir, exist_ok=True)

    latest_checkpoint = None

    start_epoch = 0

    checkpoint_status = mo.status.loading("No checkpoint found.")


    # Find latest checkpoint file

    try:

        checkpoint_files = [f for f in os.listdir(chkpt_dir) if f.endswith(".pth")]

        if checkpoint_files:

            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) # Sort by epoch number

            latest_checkpoint = os.path.join(chkpt_dir, checkpoint_files[-1])

    except OSError as e:

         print(f"Could not access checkpoint directory {chkpt_dir}: {e}")



    if load_checkpoint_flag.value and latest_checkpoint:

        try:

            print(f"Loading checkpoint: {latest_checkpoint}")

            checkpoint = torch.load(latest_checkpoint, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint['epoch'] + 1 # Start from next epoch

            # Ensure LR matches current setting if desired, or use saved LR

            # optimizer.param_groups[0]['lr'] = learning_rate.value

            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")

            checkpoint_status = mo.status.success(f"Loaded checkpoint from epoch {start_epoch - 1}.")

        except Exception as e:

            print(f"Error loading checkpoint: {e}. Starting from scratch.")

            start_epoch = 0

            checkpoint_status = mo.status.error(f"Error loading checkpoint: {e}")

            load_checkpoint_flag.value = False # Disable flag if loading failed

    elif load_checkpoint_flag.value and not latest_checkpoint:

        print("Load checkpoint selected, but no checkpoint found.")

        checkpoint_status = mo.status.warn("Load requested, but no checkpoint found.")

        start_epoch = 0

    else:

        print("Starting training from scratch.")

        checkpoint_status = mo.status.info("Starting training from scratch.")

        start_epoch = 0


    # Store training state (e.g., losses) - use Marimo state if needed, or simple list

    training_losses = []

    if load_checkpoint_flag.value and latest_checkpoint and 'losses' in checkpoint:

        training_losses = checkpoint.get('losses', []) # Load losses if saved


    # Display setup status

    setup_status = mo.vstack([

        device_info,
        model_instance_info,

        mo.md(f"Optimizer: AdamW, LR: {learning_rate.value}"),

        mo.md(f"Loss Function: MSELoss"),

        checkpoint_status

    ])
    return (

        checkpoint_status,

        chkpt_dir,

        device,

        device_info,

        latest_checkpoint,

        load_checkpoint_flag, # Pass the checkbox state itself
        loss_fn,
        model,
        model_instance_info,
        model_param_count,
        optim,

        optimizer,
        os,
        setup_status,
        start_epoch,
        torch,

        training_losses,
    )



@app.cell

def __(mo, setup_status):

    mo.md("Training setup complete.")

    mo.box(setup_status)
    return



@app.cell

def __(mo):

    mo.md("## 7. Training Loop")
    return



@app.cell
def __(

    T,

    batch_size,

    chkpt_dir,

    device,

    extract,

    get_dataloader,

    load_checkpoint_flag,
    loss_fn,
    math,
    mo,
    model,
    n_epochs,
    optim,

    optimizer,
    os,
    p_sample_loop_ddim,
    p_sample_loop_ddpm,
    plt,

    q_sample,

    save_freq_epochs,
    start_epoch,

    start_training_button,
    time,
    timesteps,
    torch,

    training_losses,

):

    # Reactive trigger: only run when the button is clicked

    _run_training = start_training_button.value


    # Use mo.state to manage the training execution and outputs persistently

    # Initialize state variables if they don't exist

    training_running = mo.state(False)

    epoch_log = mo.state([]) # Store tuples of (epoch, avg_loss)

    loss_plot_output = mo.state(None)

    status_output = mo.state(mo.md("Training has not started."))


    # Function to update the loss plot

    def update_loss_plot():

        if not epoch_log.value:

            return None

        epochs, losses = zip(*epoch_log.value)

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(epochs, losses, marker='o', linestyle='-')

        ax.set_xlabel("Epoch")

        ax.set_ylabel("Average Loss")

        ax.set_title("Training Loss Over Epochs")

        ax.grid(True)

        plt.tight_layout()

        # Capture plot to display in Marimo

        with mo.capture_figure() as fig_output:
             pass

        plt.close(fig) # Close plot to prevent display in non-Marimo contexts

        return fig_output



    if _run_training > 0 and not training_running.value:

        training_running.set(True)

        status_output.set(mo.md("Starting training..."))


        # --- Prepare DataLoader ---

        # Must be created here to use the current batch_size value

        current_batch_size = batch_size.value

        dataloader = get_dataloader(current_batch_size)

        steps_per_epoch = len(dataloader)


        # --- Load previous losses if resuming ---

        # Combine loaded losses with potentially new epoch_log structure

        # Ensure epoch numbers are consistent

        if load_checkpoint_flag.value and training_losses and not epoch_log.value:

             # Assuming training_losses is a simple list of batch losses needs adjustment

             # If checkpoint saved epoch losses, use that. Assume simple list for now.

             # This part needs refinement based on how losses are saved in checkpoint.

             # Let's reset epoch_log if checkpoint doesn't store epoch-wise avg loss.

             print("Loaded raw loss list, but can't reconstruct epoch averages accurately without epoch info per loss. Resetting plot data.")

             # Or, if checkpoint saved epoch_log format:

             # if 'epoch_log' in checkpoint: epoch_log.set(checkpoint['epoch_log'])


        # --- Training Loop ---
        model.train()

        total_start_time = time.time()


        for epoch in range(start_epoch, n_epochs.value):

            epoch_start_time = time.time()

            epoch_loss = 0.0

            status_output.set(mo.status.progress(title=f"Epoch {epoch+1}/{n_epochs.value}", subtitle="Starting...", value=0))


            for step, batch in enumerate(dataloader):

                optimizer.zero_grad()


                batch = batch.to(device)

                b_size = batch.shape[0]


                # 1. Sample random timesteps t for each image in the batch

                t = torch.randint(0, T, (b_size,), device=device).long()


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

                progress_val = (step + 1) / steps_per_epoch

                current_loss = loss.item()

                status_output.set(mo.status.progress(

                    title=f"Epoch {epoch+1}/{n_epochs.value}",

                    subtitle=f"Step {step+1}/{steps_per_epoch} | Batch Loss: {current_loss:.4f}",

                    value=progress_val)
                )


            # --- End of Epoch ---

            avg_epoch_loss = epoch_loss / steps_per_epoch

            epoch_end_time = time.time()

            epoch_duration = epoch_end_time - epoch_start_time


            # Log epoch loss

            new_log = epoch_log.value + [(epoch + 1, avg_epoch_loss)]

            epoch_log.set(new_log) # Update Marimo state


            # Update plot
            loss_plot_output.set(update_loss_plot())


            status_output.set(mo.md(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s. Average Loss: {avg_epoch_loss:.6f}"))

            print(f"Epoch {epoch+1}/{n_epochs.value} | Avg Loss: {avg_epoch_loss:.6f} | Time: {epoch_duration:.2f}s")


            # --- Checkpointing ---

            if (epoch + 1) % save_freq_epochs.value == 0 or (epoch + 1) == n_epochs.value:

                checkpoint_path = os.path.join(chkpt_dir, f"diffusion_model_epoch_{epoch+1}.pth")

                try:

                    torch.save({

                        'epoch': epoch,

                        'model_state_dict': model.state_dict(),

                        'optimizer_state_dict': optimizer.state_dict(),

                        'loss': avg_epoch_loss,

                        # Save epoch log for plotting continuity

                        'epoch_log': epoch_log.value,

                        # You might want to save other things like config, learning rate etc.

                    }, checkpoint_path)

                    print(f"Checkpoint saved to {checkpoint_path}")

                except Exception as e:

                    print(f"Error saving checkpoint: {e}")


        # --- Training Finished ---

        total_end_time = time.time()

        total_duration = total_end_time - total_start_time

        final_status = f"Training finished after {n_epochs.value} epochs ({total_duration:.2f}s)."
        status_output.set(mo.md(final_status))
        print(final_status)

        training_running.set(False) # Allow starting again


    # Display outputs

    mo.vstack([

        status_output.value,

        loss_plot_output.value if loss_plot_output.value else mo.md("Loss plot will appear here after the first epoch.")

    ])
    return (

        epoch_log,
        loss_plot_output,
        status_output,

        training_running,
        update_loss_plot,
    )



@app.cell

def __(mo):

    mo.md("## 8. Sampling / Inference")
    return



@app.cell
def __(

    BytesIO,

    T,
    channels,
    ddim_eta,

    device,

    generate_samples_button,

    image_size,

    make_grid,
    mo,
    model,

    num_samples_to_generate,
    p_sample_loop_ddim,
    p_sample_loop_ddpm,
    sampler_choice,
    timesteps,
    torch,

    transform, # Need the inverse transform

):

    # State for generated samples

    generated_samples_output = mo.state(mo.md("Click 'Generate Samples' to generate images using the trained model."))


    # Inverse transform: [-1, 1] -> [0, 1] -> [0, 255] -> PIL Image

    def tensor_to_pil(img_tensor):

        img_tensor = (img_tensor + 1) / 2 # [-1, 1] -> [0, 1]

        img_tensor = img_tensor.clamp(0, 1) # Ensure range

        # If using make_grid, it handles multiple images. If single image:

        # img_tensor = img_tensor.squeeze(0) # Remove batch dim if present

        pil_img = T.ToPILImage()(img_tensor.cpu())

        return pil_img


    if generate_samples_button.value > 0:

        generated_samples_output.set(mo.md("Generating samples..."))

        model.eval() # Set model to evaluation mode


        num_samples = num_samples_to_generate.value

        sample_shape = (num_samples, channels, image_size.value, image_size.value)


        # Select sampler

        if sampler_choice.value == "ddpm":

            generated_samples_output.set(mo.md(f"Generating {num_samples} samples using DDPM (T={T})..."))

            # DDPM needs full T steps, might be slow

            samples_list = p_sample_loop_ddpm(model, sample_shape, device)

            final_samples = samples_list[-1] # Get the last step (fully denoised)

        elif sampler_choice.value == "ddim":

            num_inference_steps = 50 # Use fewer steps for DDIM speedup

            eta = ddim_eta.value

            generated_samples_output.set(mo.md(f"Generating {num_samples} samples using DDIM (Steps={num_inference_steps}, Eta={eta})..."))

            samples_list = p_sample_loop_ddim(model, sample_shape, device, num_inference_steps=num_inference_steps, eta=eta)

            final_samples = samples_list[-1]

        else:

            final_samples = None

            generated_samples_output.set(mo.md("Invalid sampler selected."))



        if final_samples is not None:

            # Create a grid

            grid = make_grid(final_samples, nrow=int(math.sqrt(num_samples)), padding=2)

            pil_grid = tensor_to_pil(grid)


            # Convert PIL image to bytes for display

            buf = BytesIO()

            pil_grid.save(buf, format='PNG')

            buf.seek(0)

            generated_samples_output.set(mo.image(buf.getvalue()))

        else:

             generated_samples_output.set(mo.md("Sample generation failed."))



    mo.vstack([

        generated_samples_output.value

    ])

    return generated_samples_output, tensor_to_pil



@app.cell

def __(mo):
    mo.md(

        """

        ## 9. Conclusion & Next Steps


        This notebook provided a basic framework for:

        1. Generating synthetic image data on-the-fly.

        2. Setting up a U-Net model for diffusion.

        3. Implementing the core DDPM forward and reverse (sampling) processes.

        4. Adding DDIM sampling as a faster alternative.

        5. Training the model on a CPU (demonstration purposes).

        6. Saving and loading checkpoints.

        7. Generating new images using the trained model.


        **Limitations & Potential Improvements:**

        *   **CPU Speed:** Training is extremely slow. Use a GPU for practical training.

        *   **Model Size:** The U-Net used is small for faster CPU execution. Real applications use larger models.

        *   **Data Quality:** The synthetic data generation is basic. More sophisticated rendering would yield better results but be slower. Using real datasets (CelebA, CIFAR-10, LSUN, etc.) is standard practice.

        *   **Diffusion Steps (T):** Reduced T for speed. Larger T (e.g., 1000) is common but slower.

        *   **Advanced Techniques:** Explore different noise schedules (cosine), improved U-Net architectures (attention), classifier-free guidance, other samplers (EDM/Karras), etc.

        *   **Hyperparameter Tuning:** Learning rate, batch size, model dimensions, T, beta schedule all affect results significantly.

        """
    )
    return



if __name__ == "__main__":
    app.run()