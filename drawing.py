from dataclasses import dataclass
from typing import Literal

import numpy as np
import tqdm
from PIL import Image, ImageDraw


@dataclass
class DrawingParams:
    # Specifying the target image
    target_image_path: str = "images/london_eye_1b.jpg"
    x: int = 1000

    # Mode: Chinese characters, or shapes
    mode: Literal["chars", "shapes"] = "chars"

    # Specifying the shape (Chinese characters)
    char_path: str = "shapes/first-shape-test.png"
    char_size: int = 25
    shape_type: Literal["circle", "rect", "tri", "hex", "arc"] = "circle"
    shape_sizes: tuple[int, int] = (50, 300)
    shape_n_sizes: int = 25
    shape_thickness: int = 1
    shape_output_thickness: int = 1
    shape_size_distn: Literal["uniform", "spike"] = "uniform"  # Edit - don't really need to use spike!

    # Specifying the drawing process
    n_shapes: int = 4000
    n_random: int = 100
    darkness: float = 0.25
    output_x: int = 4000
    negative_penalty = 1.0  # If > 0.0 then we treat differently to just averaging values

    def __post_init__(self):
        # Load in image
        self.target_image = Image.open(self.target_image_path).convert("L")
        width, height = self.target_image.size

        self.y = int(self.x * height / width)
        self.output_sf = self.output_x / self.x  # Scale factor for output size
        self.output_y = int(self.y * self.output_sf)

        self.target_image = self.target_image.resize((self.x, self.y))
        self.target_image_array = 1 - np.array(self.target_image) / 255

        assert self.shape_n_sizes % 2 == 1, "Should be an odd number of sizes"

    def get_drawing_coords(self, sizes: list[int], output: bool = False) -> list[np.ndarray]:
        """
        Returns coordinates of black pixels in the image.

        Args:
            sizes: List of sizes of each shape / character
            output: If True, thickness will be given by `shape_output_thickness` instead of `shape_thickness`.

        Returns:
            List of arrays each of shape (2, n_pixels) representing the black pixels in
            the image. For Chinese characters this is taken from the loaded image; for
            shapes this is calculated directly.
        """
        coords_list = []

        for size in sizes:
            size = int(size)
            if self.mode == "chars":
                img = Image.open(self.char_path).convert("L")
                img = img.resize((size, size))
                img_array = np.array(img)
                black_pixels = img_array < 245  # Threshold for "black" ?
                y_coords, x_coords = np.where(black_pixels)
                coords = np.stack([y_coords, x_coords])
                coords_list.append(coords)
            else:
                t = self.shape_output_thickness if output else self.shape_thickness
                if self.shape_type == "circle":
                    img = Image.new("L", (size * 2, size * 2), 255)
                    draw = ImageDraw.Draw(img)
                    draw.ellipse([t, t, size * 2 - t, size * 2 - t], outline=0, width=t)
                elif self.shape_type == "rect":
                    img = Image.new("L", (size, size), 255)
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([t, t, size - t, size - t], outline=0, width=t)
                elif self.shape_type == "tri":
                    img = Image.new("L", (size, size), 255)
                    draw = ImageDraw.Draw(img)
                    h = int(size * np.sqrt(3) / 2)
                    points = [(0, size - 1), (size - 1, size - 1), (size // 2, size - h)]
                    draw.polygon(points, outline=0, width=t)
                elif self.shape_type == "hex":
                    img = Image.new("L", (size, size), 255)
                    draw = ImageDraw.Draw(img)
                    h = int(size * np.sqrt(3) / 4)
                    points = [
                        (0, size // 2),
                        (size // 4, size - 1),
                        (3 * size // 4, size - 1),
                        (size - 1, size // 2),
                        (3 * size // 4, size // 2 - h),
                        (size // 4, size // 2 - h),
                    ]
                    draw.polygon(points, outline=0, width=t)
                elif self.shape_type == "arc":
                    raise NotImplementedError("Not implemented these shapes yet.")

                img_array = np.array(img)
                black_pixels = img_array < 128
                y_coords, x_coords = np.where(black_pixels)
                coords = np.stack([y_coords, x_coords])
                coords_list.append(coords)

        return coords_list

    def create_img(self):
        # Create a blank canvas which we draw on with the character shape
        canvas = np.ones((self.output_y, self.output_x), dtype=np.uint8) * 255

        # Start with a version of the target image that we will draw on
        target_image = self.target_image_array.copy()

        # Create the list of coordinates - for Chinese characters this will just give us
        # a single coords array, but for shapes this will be a range of sizes.
        if self.mode == "chars":
            sizes = [self.char_size]
        else:
            sizes = np.linspace(*self.shape_sizes, num=self.shape_n_sizes, endpoint=True).tolist()

        coords = self.get_drawing_coords(sizes)
        coords_large = self.get_drawing_coords([size * self.output_sf for size in sizes], output=True)

        max_n_pixels = max(x.shape[-1] for x in coords)

        # Draw the lines
        pbar = tqdm.tqdm(range(self.n_shapes), desc="Drawing shapes")
        for step in pbar:
            # Randomly choose a series of `n_random` x and y coords
            random_y = np.random.randint(-self.char_size, self.y, size=self.n_random)
            random_x = np.random.randint(-self.char_size, self.x, size=self.n_random)
            random_yx = np.stack([random_y, random_x], -1)  # (n_rand, 2)

            # Randomly choose a series of sizes, get the corresponding coords (masked if out of bounds)
            if self.shape_size_distn == "uniform":
                p = None
            else:
                p = np.arange(len(coords) // 2).tolist() + np.arange(len(coords) // 2 + 1).tolist()[::-1]
                p = np.array(p) / np.array(p).sum()
            random_sizes = np.random.choice(np.arange(len(coords)), size=self.n_random, p=p)

            random_coords = [coords[i] + yx[:, None] for i, yx in zip(random_sizes, random_yx)]
            random_coords = [mask_coords(c, self.y, self.x) for c in random_coords]

            # Convert the coords to a padded array for concatenation (so we can vectorize operations)
            random_coords_arr = np.stack([pad_to_length(c, max_n_pixels) for c in random_coords])  # (n_rand, 2, n_pix)
            random_coords_mask = np.any(random_coords_arr != 0, axis=1)  # (n_rand, n_pix)

            # Get the darkness of target image of the pixels at these coords
            pixel_values = target_image[random_coords_arr[:, 0], random_coords_arr[:, 1]].copy()  # (n_rand, n_pix)

            # Process pixel values by applying negative penalty and masking, then average over each random choice
            if self.negative_penalty > 0.0:
                pixel_values[pixel_values < 0.0] *= 1 + self.negative_penalty
            pixel_values = (pixel_values * random_coords_mask).sum(-1) / (random_coords_mask.sum(-1) + 1e-8)

            # Pick the darkest shape to draw
            best_idx = np.argmax(pixel_values)
            best_yx = random_yx[best_idx]
            best_yx_large = (best_yx.astype(np.float32) * self.output_sf).astype(np.int16)
            best_size = random_sizes[best_idx].item()

            # Update the canvas, and the target image
            coords_large_best = best_yx_large[:, None] + coords_large[best_size]
            coords_large_best = mask_coords(coords_large_best, self.output_y, self.output_x, remove=True)
            canvas[*coords_large_best] = 0
            target_image[*random_coords[best_idx]] -= self.darkness

            # Update pbar with stats
            if step % 10 == 0:
                pbar.set_postfix(
                    {
                        "filled_fraction": (canvas == 0).astype(float).mean().item(),
                        # "filled_pixels": (canvas == 0).sum().item(),
                        # "avg_canvas_value": canvas.sum().item() / canvas.size,
                        "avg_target_img_value": target_image.mean().item(),
                        "best_avg_pixel_value": pixel_values[best_idx].item(),
                    }
                )

        # Display the canvas inline
        # Image.fromarray(canvas).show()
        return canvas


# TODO - can I do the code below faster?


def mask_coords(coords: np.ndarray, max_y: int, max_x: int, remove: bool = False) -> np.ndarray:
    """Masks coordinates that go out of bounds."""
    assert coords.shape[0] == 2, "Coords should have shape (2, n_pixels)"
    out_of_bounds = (coords[0] < 0) | (coords[0] >= max_y) | (coords[1] < 0) | (coords[1] >= max_x)
    if remove:
        coords = coords[:, ~out_of_bounds]
    else:
        coords[:, out_of_bounds] = 0
    return coords


def pad_to_length(arr: np.ndarray, length: int, axis: int = -1, fill_value: float = 0):
    target_shape = list(arr.shape)
    assert length >= target_shape[axis]
    target_shape[axis] = length - target_shape[axis]
    return np.concat([arr, np.full_like(arr, fill_value=fill_value, shape=tuple(target_shape))], axis=axis)
