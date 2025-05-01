import random
from typing import Any

import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray

import PIL
from PIL import Image, ImageDraw

type NDArrayImg = NDArray[Any]
type Img = PIL.Image.Image
type Draw = PIL.ImageDraw.ImageDraw


def create_img(width: int, height: int) -> Img:
    """Creates a blank image with white background."""
    img = Image.new("L", (width, height), 0)  # black background
    draw = ImageDraw.Draw(img)
    return img, draw


class Bike:
    def __init__(self, width: int, street_y_for_bg: int):
        min_radius = max(5, int(width * 0.04))
        max_radius = max(10, int(width * 0.1))
        self.wheel_radius = random.randint(min_radius, max_radius)
        self.color = random.randint(10, 255)
        self.width_approx = self.wheel_radius * 3.5 + 2 * self.wheel_radius
        self.margin_x = int(self.width_approx / 2) + 5
        self.center_x = random.randint(self.margin_x, width - self.margin_x)
        self.center_y = street_y_for_bg - random.randint(
            0, self.wheel_radius // 2
        )  # slightly above or on the line
        self.street_y_for_bg = street_y_for_bg

    def set_center_x(self, center_x: int) -> None:
        self.center_x = center_x

    def set_center_y(self, center_y: int) -> None:
        self.center_y = center_y


def add_noise(image_array: NDArrayImg, amount=0.02) -> NDArrayImg:
    """Adds salt and pepper noise to a numpy image array."""
    noisy_image = np.copy(image_array)
    num_pixels = image_array.size
    # Salt noise
    num_salt = np.ceil(amount * num_pixels * 0.5)
    coords = tuple(
        np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape
    )
    noisy_image[coords] = 255
    # Pepper noise
    num_pepper = np.ceil(amount * num_pixels * 0.5)
    coords = tuple(
        np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape
    )
    noisy_image[coords] = 0
    return noisy_image


def draw_street(draw: Draw, width: int, height: int) -> int:
    """Draws a simple urban background sketch."""
    line_color = random.randint(20, 75)  # Varying shades of grey for background
    line_thickness = 1

    street_y = int(height * random.uniform(0.7, 0.85))  # Random street level
    draw.line(
        [(0, street_y), (width, street_y)], fill=line_color, width=line_thickness + 1
    )
    trottoir_y = street_y + random.randint(3, 6)
    if trottoir_y < height:
        draw.line(
            [(0, trottoir_y), (width, trottoir_y)],
            fill=line_color,
            width=line_thickness,
        )

    return street_y


def draw_urban_background(draw: Draw, width: int, height: int, street_y: int) -> None:
    # Buildings
    current_x = 0
    line_thickness = 1
    min_building_width = max(15, int(width * 0.1))
    max_building_width = max(30, int(width * 0.3))

    while current_x < width - min_building_width:
        line_color = random.randint(50, 120)  # Varying shades of grey for background
        fill_color = random.randint(45, line_color)  # Same for buildings
        building_width = random.randint(min_building_width, max_building_width)
        building_height = random.randint(
            int(height * 0.2), street_y - 10
        )  # Ensure below top & above street
        building_top = street_y - building_height
        draw.rectangle(
            [(current_x, building_top), (current_x + building_width, street_y)],
            outline=line_color,
            width=line_thickness,
            fill=fill_color,
        )
        # Simple windows (optional)
        if random.random() < 0.7:
            win_size = max(2, int(building_width * 0.1))
            for wx in range(
                current_x + 5,
                current_x + building_width - win_size - 2,
                win_size * 2 + 3,
            ):
                for wy in range(
                    building_top + 5, street_y - win_size - 2, win_size * 2 + 3
                ):
                    if (
                        wy + win_size < street_y
                        and wx + win_size < current_x + building_width
                    ):
                        draw.rectangle(
                            [(wx, wy), (wx + win_size, wy + win_size)],
                            outline=line_color,
                            width=1,
                        )

        current_x += building_width + random.randint(0, 5)

    # Lampposts
    num_lampposts = random.randint(1, 4)
    lamp_height = random.randint(int(height * 0.15), int(height * 0.3))
    lamp_bottom = street_y + 2
    line_color = random.randint(50, 75)  # pole color
    fill_color = random.randint(50, 255)  # bulb color
    for _ in range(num_lampposts):
        lamp_x = random.randint(10, width - 10)
        lamp_top = max(5, lamp_bottom - lamp_height)
        draw.line(
            [(lamp_x, lamp_bottom), (lamp_x, lamp_top)],
            fill=line_color,
            width=line_thickness,
        )
        draw.ellipse(
            [(lamp_x - 2, lamp_top - 3), (lamp_x + 2, lamp_top)], fill=fill_color
        )


def draw_sun(draw: Draw, width: int, height: int, street_y: int) -> None:
    """Draws a simple sun sketch in the sky."""
    sun_color = random.randint(180, 255)
    sun_radius = random.randint(max(5, int(width * 0.03)), max(15, int(width * 0.08)))
    # Position in the sky (above buildings/street)
    max_y_for_sun = street_y - int(
        height * 0.1
    )  # Ensure some space above street/buildings
    if max_y_for_sun <= sun_radius * 2:
        max_y_for_sun = sun_radius * 2 + 5  # Avoid edge case

    sun_cx = random.randint(sun_radius, width - sun_radius)
    sun_cy = random.randint(
        sun_radius, max(sun_radius + 1, max_y_for_sun - sun_radius)
    )  # Place in upper part

    # Draw filled sun circle ONLY on the input image drawing context
    draw.ellipse(
        (
            sun_cx - sun_radius,
            sun_cy - sun_radius,
            sun_cx + sun_radius,
            sun_cy + sun_radius,
        ),
        fill=sun_color,
    )


def draw_bike_simplified(
    draw_sketch: Draw, draw_mask: Draw, bike: Bike, line_thickness=1
):
    """Draws the simplified bike on the sketch (input) and mask (target)."""

    wheel_radius, center_x, center_y, bike_color = (
        bike.wheel_radius,
        bike.center_x,
        bike.center_y,
        bike.color,
    )
    mask_color = 255  # Mask is always white for the bike
    mask_line_thickness = max(
        2, line_thickness * 2, wheel_radius // 4
    )  # Thicker for mask coverage

    # Simplified fixed geometry relative to wheel radius
    wheel_distance = int(wheel_radius * 3.2)
    rear_wheel_cx = center_x - wheel_distance // 2
    rear_wheel_cy = center_y
    front_wheel_cx = center_x + wheel_distance // 2
    front_wheel_cy = center_y

    A = (rear_wheel_cx, rear_wheel_cy)  # Rear wheel center
    E = (front_wheel_cx, front_wheel_cy)  # Front wheel center
    B = (center_x, center_y + wheel_radius * 0.3)  # Bottom bracket approx
    C = (
        rear_wheel_cx + wheel_radius * 0.4,
        center_y - wheel_radius * 1.8,
    )  # Seat approx
    D = (
        front_wheel_cx - wheel_radius * 0.2,
        center_y - wheel_radius * 1.5,
    )  # Handlebar approx

    # --- Draw on Sketch (Input Image) ---
    # Wheels (thin outline)
    draw_sketch.ellipse(
        (
            A[0] - wheel_radius,
            A[1] - wheel_radius,
            A[0] + wheel_radius,
            A[1] + wheel_radius,
        ),
        outline=bike_color,
        width=line_thickness,
    )
    draw_sketch.ellipse(
        (
            E[0] - wheel_radius,
            E[1] - wheel_radius,
            E[0] + wheel_radius,
            E[1] + wheel_radius,
        ),
        outline=bike_color,
        width=line_thickness,
    )
    # Frame (thin lines)
    draw_sketch.line([A, B], fill=bike_color, width=line_thickness)  # Chainstay
    draw_sketch.line([B, C], fill=bike_color, width=line_thickness)  # Seat tube
    draw_sketch.line([C, A], fill=bike_color, width=line_thickness)  # Seat stay
    draw_sketch.line([C, D], fill=bike_color, width=line_thickness)  # Top tube
    draw_sketch.line([B, D], fill=bike_color, width=line_thickness)  # Down tube
    draw_sketch.line([D, E], fill=bike_color, width=line_thickness)  # Fork

    # --- Draw on Mask (Target Image) ---
    # Wheels (filled circles)
    draw_mask.ellipse(
        (
            A[0] - wheel_radius,
            A[1] - wheel_radius,
            A[0] + wheel_radius,
            A[1] + wheel_radius,
        ),
        fill=mask_color,
    )
    draw_mask.ellipse(
        (
            E[0] - wheel_radius,
            E[1] - wheel_radius,
            E[0] + wheel_radius,
            E[1] + wheel_radius,
        ),
        fill=mask_color,
    )
    # Frame (thick lines for coverage)
    draw_mask.line([A, B], fill=mask_color, width=mask_line_thickness)
    draw_mask.line([B, C], fill=mask_color, width=mask_line_thickness)
    draw_mask.line([C, A], fill=mask_color, width=mask_line_thickness)
    draw_mask.line([C, D], fill=mask_color, width=mask_line_thickness)
    draw_mask.line([B, D], fill=mask_color, width=mask_line_thickness)
    draw_mask.line([D, E], fill=mask_color, width=mask_line_thickness)


def generate_background(width: int, height: int) -> tuple[Img, Draw, int]:
    """Generates a simple urban background."""
    # Create a blank image with white background
    img, draw = create_img(width, height)

    # Draw street and buildings
    street_y = draw_street(draw, width, height)
    draw_sun(draw, width, height, street_y)
    draw_urban_background(draw, width, height, street_y)

    return img, draw, street_y


def add_bike(
    draw: Draw,
    width: int,
    height: int,
    bike_x: int,
    bike_y: int,
    street_y: int,
    bike: Bike,
) -> tuple[Img, Draw]:
    """Adds a bike to the background."""
    mask_img, mask_draw = create_img(width, height)

    if bike_x >= 0:
        bike.set_center_x(bike_x)

    if bike_y >= 0:
        bike.set_center_y(bike_y)

    # Draw the bike on the sketch and mask
    draw_bike_simplified(draw, mask_draw, bike)

    return mask_img, mask_draw


def show_two_pictures(
    img1: Img,
    txt1: str,
    img2: Img,
    txt2: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Displays two images side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img1, cmap="gray")
    ax2.imshow(img2, cmap="gray")
    ax1.axis("off")
    ax2.axis("off")
    ax1.set_title(txt1)
    ax2.set_title(txt2)
    plt.tight_layout()
    return plt.gcf(), plt.gca()


def generate_bike_sample(
    width: int, height: int, add_noise_prob=0.5, noise_amount=0.04
) -> tuple[NDArrayImg, NDArrayImg]:
    """Generates a bike sample with a background."""
    # Generate the background
    img, draw, street_y = generate_background(width, height)

    # Create a bike instance
    bike = Bike(width, street_y)

    # Add the bike to the background
    mask_img, mask_draw = add_bike(draw, width, height, -1, -1, street_y, bike)

    img_np = np.array(img)
    mask_np = np.array(mask_img)

    if random.random() < add_noise_prob:
        img_np = add_noise(img_np, amount=noise_amount)

    return img_np, mask_np


def mask_to_bbox(mask_np: NDArrayImg, threshold=0.5):
    """Calculates the bounding box from a binary mask (NumPy array)."""
    # Ensure mask is binary
    binary_mask = (mask_np > threshold).astype(np.uint8)

    # Find contours - OpenCV is good for this, but let's use NumPy for simplicity
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None  # No object found

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Return format: (xmin, ymin, xmax, ymax) consistent with PIL draw
    return (cmin, rmin, cmax, rmax)


def draw_bbox(image_pil, bbox, color="red", thickness=2):
    """Draws a bounding box on a PIL image."""
    if bbox is None:
        return image_pil  # Return original if no bbox

    draw = ImageDraw.Draw(image_pil)
    # The bbox tuple is (xmin, ymin, xmax, ymax)
    draw.rectangle(bbox, outline=color, width=thickness)
    return image_pil


def main(img_size: int) -> plt.Figure:
    # Let's test the generated image
    width, height = img_size, img_size
    test_img_np, test_mask_np = generate_bike_sample(width, height)
    # Generate a bike sample
    img, mask = generate_bike_sample(
        img_size, img_size, add_noise_prob=0.5, noise_amount=0.04
    )
    # Display the images
    

    print(f"Input shape: {test_img_np.shape}, Mask shape: {test_mask_np.shape}")
    print(f"Input dtype: {test_img_np.dtype}, Mask dtype: {test_mask_np.dtype}")
    print(f"Input min/max: {np.min(test_img_np)}/{np.max(test_img_np)}")
    print(f"Mask min/max: {np.min(test_mask_np)}/{np.max(test_mask_np)}")

    fig, axes = show_two_pictures(
        test_img_np, "Generated Input Image", test_mask_np, "Generated Mask"
    )
    return fig

if __name__ == "__main__":
    fig = main(128)  # Test with a 128x128 image
    plt.show()
    plt.close(fig)  # Close the figure after showing it
