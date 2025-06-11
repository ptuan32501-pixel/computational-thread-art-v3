import enum
import math
import pprint
import time
from calendar import c
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import tqdm
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image, ImageDraw

from image_color import blur_image
from misc import get_color_hash, get_img_hash

Arr = np.ndarray


class LineType(enum.Enum):
    STRAIGHT = enum.auto()
    ARC = enum.auto()
    BEZIER = enum.auto()


class ShapeType(enum.Enum):
    CIRCLE = enum.auto()
    RECT = enum.auto()
    TRI = enum.auto()
    HEX = enum.auto()


@dataclass
class Shape:
    """
    Most general class for shapes: contains characters, connected arcs, and all uniform shapes.
    """

    line_type: LineType | None = None
    shape_type: ShapeType | None = None
    character_path: str | None = None

    # Not parameter-specific
    max_out_of_bounds: int | None = None  # No lines are allowed to go more than this far out of bounds, at any point

    # * Used for shapes and lines (not characters)
    size_range: tuple[float, float] = (0.05, 0.15)  # size range (means smth different for arcs)

    # * Only used for lines
    endpoint_angle_range: tuple[float, float] = (np.pi * 0.2, np.pi * 0.6)
    # ^ absolute value of angle change (shouldn't be zero because then arc is just a straight line)
    bezier_control_factor_range: tuple[float, float] = (0.4, 0.6)
    # ^ not exactly sure yet what effect this has
    bezier_end_angle_range: tuple[float, float] = (-np.pi * 0.6, np.pi * 0.6)
    # ^ direction we end by pointing in (relative to what it would be if we just went straight)

    def __post_init__(self):
        assert sum(x is not None for x in [self.shape_type, self.character_path, self.line_type]) == 1, (
            "Must specify one of shape_type, character_path, or line_type"
        )
        if self.line_type is not None:
            assert self.shape_type is None, "Can't specify both line_type and shape_type"
            assert self.character_path is None, "Can't specify both line_type and character_path"

        if self.line_type is not None:
            pass  # TODO - add in appropriate restrictions here

    def get_random_params(
        self,
        start_coords: Float[Arr, "2"],
        start_dir: Float[Arr, "2"],
        canvas_length: float,
    ) -> dict:
        """Gets random parameters for the shape."""

        # Chinese characters have no free parameters except for position
        if self.character_path is not None:
            return {}

        # Non-line shapes have a single free parameter: size
        elif self.shape_type is not None:
            return {"size": np.random.randint(*self.size_range)}

        # Arcs have many free parameters!
        else:
            # Get angles. The end angle represents the angle we're moving in the direction of, if we moved
            # straight from start to end (which is why it's close to `start_angle`).
            start_angle = np.arctan2(*start_dir)
            angle_delta = np.random.uniform(*self.endpoint_angle_range) * (1 if np.random.rand() < 0.5 else -1)
            end_angle = start_angle + angle_delta

            # Get distances / end points.
            radius = np.random.rand() * (self.size_range[1] - self.size_range[0]) + self.size_range[0]
            radius *= canvas_length
            end_coords_delta = radius * np.array([np.sin(end_angle), np.cos(end_angle)])
            end_coords = start_coords + end_coords_delta

            if self.line_type == LineType.STRAIGHT:
                return {"end_coords": end_coords}
            elif self.line_type == LineType.ARC:
                return {
                    "end_coords": end_coords,
                    "angle_delta": angle_delta,  # Not needed, just useful for getting end direction
                }
            elif self.line_type == LineType.BEZIER:
                end_angle_direction = end_angle + np.random.uniform(*self.bezier_end_angle_range)
                return {
                    "end_coords": end_coords,
                    "end_dir": np.array([np.sin(end_angle_direction), np.cos(end_angle_direction)]),
                    "start_strength": np.random.uniform(*self.bezier_control_factor_range),
                    "end_strength": np.random.uniform(*self.bezier_control_factor_range),
                }

    def get_drawing_coords_list(
        self,
        n_shapes: int,
        start_dir: Float[Arr, "2"],
        start_coords: Float[Arr, "2"],
        canvas_y: int,
        canvas_x: int,
        outer_bound: float | None,
        inner_bound: float | None,
    ) -> list[tuple[dict, Float[Arr, "2 n_pixels"], Float[Arr, "2 n_pixels"], Float[Arr, "2"]]]:
        """Generates `n_shapes` random shapes, and returns a list of their coords.

        Args:
            n_shapes: Number of shapes to generate
            start_dir: Direction to start the shape in
            start_coords: Coordinates to start the shape at
            canvas_y: Height of the canvas
            canvas_x: Width of the canvas
            outer_bound: Max value out of bounds we allow ANY pixel to be
            inner_bound: Min negative value out of bounds we allow the LAST pixel to be

        Returns:
            List of the following:
            - params: Dictionary of parameters used to generate the shape
            - coords: (2, n_pixels) array of float coordinates for the shape
            - coords_uncropped: (2, n_pixels) without cropping at sides (useful for final drawing)
            - pixels: (2, n_pixels) array of integer (y, x) coordinates for the shape
        """
        canvas_length = min(canvas_x, canvas_y)

        if self.line_type is None:
            raise NotImplementedError("Only doing lines for now (take commented code at the end)")

        coords_list = []

        while len(coords_list) < n_shapes:
            # Get random parameterization for this shape
            params = self.get_random_params(start_coords, start_dir, canvas_length)

            # We ignore the start coords and randomize them, if this isn't an arc
            if self.line_type is None:
                start_coords = np.random.rand(2) * np.array([canvas_x, canvas_y])

            # Get the actual coordinates for this shape
            coords_uncropped, _, end_dir = self.draw_curve(start_coords, start_dir, **params)

            # Crop parts of `coords` which go off the edge, and only keep ones which are in bounds anywhere
            coords = mask_coords(coords_uncropped, canvas_y, canvas_x, outer_bound, inner_bound, remove=True)
            if coords.shape[-1] > 0:
                coords_list.append((params, coords, coords_uncropped, end_dir))

        return coords_list

    def draw_curve(
        self,
        start_coords: Float[Arr, "2"],
        start_dir: Float[Arr, "2"],
        end_coords: Float[Arr, "2"],
        **kwargs,
    ) -> tuple[Float[Arr, "2 n_pixels"], Int[Arr, "2 n_pixels"], Float[Arr, "2"]]:
        """
        Draw arc or Bezier curve and return interpolated pixel coordinates.

        Args:
            start_coords: (y, x) starting coordinates as floats
            start_dir: normalized (y, x) direction vector
            **kwargs: For 'arc': radius, angle, orientation (True=anticlockwise)
                    For 'bezier': end_coords, start_length, end_length, end_dir

        Returns:
            coords: (2, num_pixels) array of float coordinates
            pixels: (2, num_pixels) array of integer (y, x) coordinates
            final_dir: (2,) array with final normalized (y, x) direction
        """
        assert self.line_type in [LineType.STRAIGHT, LineType.ARC, LineType.BEZIER]

        # We start by getting 3 things:
        # - Length of curve
        # - Final direction
        # - Function which interpolates the curve in the region [0, 1]

        if self.line_type == LineType.STRAIGHT:
            assert set(kwargs.keys()) == set()

            line_length = np.linalg.norm(end_coords - start_coords)

            interpolate = lambda t: start_coords[:, None] + t * (end_coords - start_coords)[:, None]
            final_dir = start_dir

        elif self.line_type == LineType.ARC:
            assert set(kwargs.keys()) == {"angle_delta"}

            raise NotImplementedError("Haven't fixed bugs in this yet.")

            # Get center of circle, by moving radially inwards from `start_coords`
            chord = end_coords - start_coords
            perp = np.array([-start_dir[1], start_dir[0]])
            radius = np.linalg.norm(chord) ** 2 / (2 * np.abs(np.dot(chord, perp)))
            center = start_coords + radius * perp

            # Use this to figure out the (signed) angle of the arc we're making around this circle
            start_vec = start_coords - center
            end_vec = end_coords - center
            start_radial_angle = np.arctan2(*start_vec)
            end_radial_angle = np.arctan2(*end_vec)
            delta_radial_angle = (end_radial_angle - start_radial_angle) % (2 * np.pi)
            if delta_radial_angle > np.pi:
                delta_radial_angle -= 2 * np.pi  # convert to clockwise if this makes a shorter trip around
            assert -np.pi <= delta_radial_angle <= np.pi, "Angle delta is out of bounds"

            # Use angle & radius to get interpolation function
            interpolate = lambda t: center[:, None] + radius * np.array(
                [
                    np.sin(start_radial_angle + t * delta_radial_angle),
                    np.cos(start_radial_angle + t * delta_radial_angle),
                ]
            )

            final_angle = np.arctan2(*start_dir) + delta_radial_angle
            final_dir = np.array([np.cos(final_angle), np.sin(final_angle)])
            line_length = np.abs(kwargs["angle_delta"]) * radius

        elif self.line_type == LineType.BEZIER:
            assert set(kwargs.keys()) == {"start_strength", "end_strength", "end_dir"}
            start_strength, end_strength, end_dir = kwargs["start_strength"], kwargs["end_strength"], kwargs["end_dir"]

            # Get chord length, to be used for computing control points
            chord = end_coords - start_coords
            chord_length = np.linalg.norm(chord)

            # Default for `end_dir` is to be pointing in the direction of the direct path from start to end
            if end_dir is None:
                end_dir = chord / chord_length

            # Control points
            p0 = np.array(start_coords)
            p1 = np.array(start_coords) + start_strength * chord_length * start_dir
            p3 = np.array(end_coords)
            p2 = np.array(end_coords) - end_strength * chord_length * end_dir

            # Estimate curve length and steps: avg of direct path and control points piecewise linear path
            chord_length = np.linalg.norm(p3 - p0)
            control_length = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
            line_length = (chord_length + control_length) / 2

            # Generate Bezier points
            interpolate = lambda t: (
                ((1 - t) ** 3) * p0[:, None]
                + (3 * (1 - t) ** 2 * t) * p1[:, None]
                + (3 * (1 - t) * t**2) * p2[:, None]
                + (t**3) * p3[:, None]
            )
            final_dir = np.array(end_dir)

        # We get coords by interpolating along the curve, a certain number of steps
        num_steps = 1 + max(1, int(line_length))
        coords = interpolate(np.linspace(0, 1, num_steps, endpoint=True))

        # Round to pixels and normalize final direction
        pixels = np.round(coords).astype(int)
        pixels = np.unique(pixels, axis=1)
        final_dir = final_dir / np.linalg.norm(final_dir)

        return coords, pixels, final_dir


# TODO - allow this code to be re-used? (it's mostly copy-pased from `image_color.py`)


@dataclass
class TargetImage:
    image_path: str
    weight_image_path: str | None
    palette: list[tuple[int, int, int]]
    x: int
    output_x: int
    blur_rad: float | None = 4
    display_dithered: bool = False

    def __post_init__(self):
        # Check colors are valid (raise error if not)
        _ = [get_color_string(color) for color in self.palette]

        # Load in image
        image = Image.open(self.image_path).convert("L" if len(self.palette) == 1 else "RGB")

        # Optionally load in (and turn to an array) the weight image
        self.weight_image = None
        if self.weight_image_path is not None:
            weight_image = Image.open(self.weight_image_path).convert("L")
            self.weight_image = np.asarray(weight_image.resize((self.x, self.y))).astype(np.float32) / 255

        # Get dimensions (for target and output images)
        width, height = image.size
        self.y = int(self.x * height / width)
        self.output_sf = self.output_x / self.x
        self.output_y = int(self.y * self.output_sf)

        # Optionally perform dithering, and get `self.image_dict` for use in `Drawing`
        image_arr = np.asarray(image.resize((self.x, self.y)))
        if len(self.palette) == 1:
            self.image_dict = {self.palette[0]: 1.0 - image_arr.astype(np.float32) / 255}
        else:
            assert (255, 255, 255) not in self.palette, "White should not be in palette"
            image_dithered = FS_dither(image_arr, [(255, 255, 255)] + self.palette)
            self.image_dict = {
                color: (get_img_hash(image_dithered) == get_color_hash(np.array(color))).astype(np.float32)
                for color in self.palette
            }
            if self.blur_rad is not None:
                self.image_dict = {color: blur_image(img, self.blur_rad) for color, img in self.image_dict.items()}

            nonwhite_density_sum = sum(
                [img.sum() for color, img in self.image_dict.items() if color != (255, 255, 255)]
            )
            for color, img in self.image_dict.items():
                print(f"{get_color_string(color)}, density = {img.sum() / nonwhite_density_sum:.4f}")

        # Display the dithered image
        if self.display_dithered:
            background_colors = [
                np.array([255, 255, 255]) if sum(color) < 255 + 160 else np.array([0, 0, 0]) for color in self.palette
            ]
            dithered_images = np.concatenate(
                [
                    bg + img[:, :, None] * (np.array(color) - bg)
                    for bg, (color, img) in zip(background_colors, self.image_dict.items())
                ],
                axis=1,
            )
            px.imshow(
                dithered_images,
                height=290,
                width=100 + 200 * len(self.palette),
                title=" | ".join([str(x) for x in self.palette]),
            ).update_layout(margin=dict(l=10, r=10, t=40, b=10)).show()


@dataclass
class Drawing:
    target: TargetImage

    shape: Shape

    n_shapes: int | list[int]
    n_random: int
    darkness: float | list[float]
    negative_penalty: float

    # Outer bound means we don't allow any lines to go further than this far out of bounds. Inner bound
    # means we don't allow any lines to END closer than this to the edge. Inner is important because without
    # it, we might finish 1 pixel away from the end and then we'd be totally fucked.
    outer_bound: float | None
    inner_bound: float | None

    def __post_init__(self):
        if self.outer_bound is not None:
            assert self.outer_bound > 0, "Outer bound must be positive"
            assert self.inner_bound is not None and self.inner_bound > 0, (
                "Inner bound must be supplied if outer bound is"
            )

    def create_img(
        self, seed: int = 0
    ) -> tuple[list[dict], Image.Image, Int[Arr, "y x"], dict[str, Float[Arr, "n_pixels 2"]]]:
        np.random.seed(seed)

        # Get our starting position & direction (posn is random, direction is pointing inwards)
        start_coords = (0.1 + 0.8 * np.random.rand(2)) * np.array([self.target.y, self.target.x])
        start_coords_offset = start_coords - np.array([self.target.y, self.target.x]) / 2
        start_dir = -start_coords_offset / (np.linalg.norm(start_coords_offset) + 1e-6)

        # If any parameters were given as a single number, convert them to lists
        if isinstance(self.darkness, float):
            self.darkness = [self.darkness] * len(self.target.palette)
        if isinstance(self.n_shapes, int):
            self.n_shapes = [self.n_shapes]
        assert len(self.n_shapes) == len(self.target.palette), "Should give num shapes for each color"

        # Create dicts to store params and coords for each color
        all_params = {}
        all_coords = {}

        for color, n_shapes, darkness in zip(self.target.palette, self.n_shapes, self.darkness, strict=True):
            if n_shapes == 0:
                continue

            image = self.target.image_dict[color]

            color_string = get_color_string(color)
            all_coords[color_string] = []
            all_params[color_string] = []
            pbar = tqdm.tqdm(range(n_shapes), desc=f"Drawing {color_string}")

            for step in pbar:
                # Get our random parameterized shapes
                coords_list = self.shape.get_drawing_coords_list(
                    n_shapes=self.n_random,
                    start_dir=start_dir,
                    start_coords=start_coords,
                    canvas_y=self.target.y,
                    canvas_x=self.target.x,
                    outer_bound=self.outer_bound,
                    inner_bound=self.inner_bound,
                )

                # Turn them into integer pixels, and concat them
                pixels = [coords.astype(np.int32) for _, coords, _, _ in coords_list]
                n_pixels = [coords.shape[-1] for _, coords, _, _ in coords_list]
                pixels = np.stack([pad_to_length(p, max(n_pixels)) for p in pixels])  # (n_rand, 2, n_pix)

                # Get the pixels values of the target image at these coords
                pixel_values = image[pixels[:, 0], pixels[:, 1]]  # (n_rand, n_pix)
                pixel_values_mask = np.any(pixels != 0, axis=1)  # (n_rand, n_pix)

                # Apply negative penalty and weighting
                if self.negative_penalty > 0.0:
                    # pixel_values[pixel_values < 0.0] *= 1 + self.negative_penalty
                    pixel_values -= self.negative_penalty * np.maximum(0.0, self.darkness - pixel_values)

                if self.target.weight_image is not None:
                    pixel_weights = self.target.weight_image[pixels[:, 0], pixels[:, 1]]  # (n_rand, n_pix)
                    pixel_values_mask = pixel_values_mask.astype(pixel_values.dtype) * pixel_weights

                # Average over each pixel array
                pixel_values = (pixel_values * pixel_values_mask).sum(-1) / (pixel_values_mask.sum(-1) + 1e-8)

                # Pick the darkest shape to draw
                best_idx = np.argmax(pixel_values)
                best_params, best_coords, best_coords_uncropped, best_end_dir = coords_list[best_idx]

                # Subtract it from the target image, and write it to the canvas
                best_pixels = best_coords.astype(np.int32)
                image[best_pixels[0], best_pixels[1]] -= darkness
                all_params[color_string].append(best_params)
                all_coords[color_string].append(best_coords_uncropped)

                # This end dir is the new start dir (same for position)
                start_dir = best_end_dir
                start_coords = best_coords[:, -1]

        # Create canvas and draw on it
        canvas = Image.new("RGB", (self.target.output_x, self.target.output_y), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for color_string, coords in all_coords.items():
            all_coords[color_string] = np.concatenate(coords, axis=1).T  # shape (n_pixels, 2)
            coords = (self.target.output_sf * all_coords[color_string]).tolist()
            for (y0, x0), (y1, x1) in zip(coords[:-1], coords[1:]):
                draw.line([(x0, y0), (x1, y1)], fill=color_string, width=1)

        return all_params, canvas, image, all_coords


def get_color_string(color: tuple[int, int, int]):
    color_string = {
        (0, 0, 0): "black",
        (0, 215, 225): "aqua",
        (0, 120, 240): "dodgerblue",
        (0, 0, 128): "darkblue",
        (255, 255, 255): "white",
        (255, 0, 0): "red",
    }.get(color, None)

    if color_string is None:
        raise ValueError(f"Color {color} not found in color string")

    return color_string


def mask_coords(
    coords: Float[Arr, "2 n_pixels"],
    max_y: int,
    max_x: int,
    outer_bound: float | None,
    inner_bound: float | None,
    remove: bool = False,
) -> Float[Arr, "2 n_pixels"]:
    """Masks coordinates that go out of bounds."""
    assert coords.shape[0] == 2, "Coords should have shape (2, n_pixels)"

    # Return empty array if either (1) ANY pixels are too far out of bounds or (2) we END too close to an edge
    max_out_of_bounds = np.max(
        [
            -coords[0].min() / max_y,
            (coords[0].max() - max_y) / max_y,
            -coords[1].min() / max_x,
            (coords[1].max() - max_x) / max_x,
        ]
    )
    if max_out_of_bounds > outer_bound:
        return coords[:, :0]

    end_out_of_bounds = np.max(
        [
            -coords[0, -1] / max_y,
            (coords[0, -1] - max_y) / max_y,
            -coords[1, -1] / max_x,
            (coords[1, -1] - max_x) / max_x,
        ]
    )
    if end_out_of_bounds > -inner_bound:
        return coords[:, :0]

    # Remove all out of bounds coordinates
    out_of_bounds = (coords[0] < 0) | (coords[0] >= max_y) | (coords[1] < 0) | (coords[1] >= max_x)
    coords = coords[:, ~out_of_bounds] if remove else np.where(out_of_bounds, 0.0, coords)
    return coords


def pad_to_length(arr: np.ndarray, length: int, axis: int = -1, fill_value: float = 0):
    target_shape = list(arr.shape)
    assert length >= target_shape[axis]
    target_shape[axis] = length - target_shape[axis]
    return np.concatenate([arr, np.full_like(arr, fill_value=fill_value, shape=tuple(target_shape))], axis=axis)


def FS_dither(
    image: Int[Arr, "y x 3"],
    palette: list[tuple[int, int, int]],
    pixels_per_batch: int = 32,
    num_overlap_rows: int = 6,
) -> tuple[Float[Arr, "y x 3"], float]:
    t0 = time.time()

    image_dithered: Float[Arr, "y x 3"] = image.astype(np.float32)
    y, x = image_dithered.shape[:2]

    num_batches = math.ceil(y / pixels_per_batch)
    rows_to_extend_by = num_batches - (y % num_batches)

    # Add a batch dimension
    image_dithered = einops.rearrange(
        np.concatenate([image_dithered, np.zeros((rows_to_extend_by, x, 3))]),
        "(batch y) x rgb -> y x batch rgb",
        batch=num_batches,
    )
    # Concat the last `num_overlap_rows` to the start of the image
    end_of_each_batch = np.concatenate(
        [np.zeros((num_overlap_rows, x, 1, 3)), image_dithered[-num_overlap_rows:, :, :-1]], axis=-2
    )
    image_dithered = np.concatenate([end_of_each_batch, image_dithered], axis=0)

    image_dithered = FS_dither_batch(image_dithered, palette)

    image_dithered = einops.rearrange(
        image_dithered[num_overlap_rows - 1 : -1],
        "y x batch rgb -> (batch y) x rgb",
    )[1 : y + 1]

    print(f"FS dithering complete in {time.time() - t0:.2f}s")

    return image_dithered


def FS_dither_batch(
    image_dithered: Float[Arr, "y x batch 3"],
    palette: list[tuple[int, int, int]],
) -> Int[Arr, "y x batch 3"]:
    # Define the constants we'll multiply with when "shifting the errors" in dithering
    AB = np.array([3, 5]) / 16
    ABC = np.array([3, 5, 1]) / 16
    BC = np.array([5, 1]) / 16

    palette = np.array(palette)  # [palette 3]

    # Set up stuff
    palette_sq = einops.rearrange(palette, "palette rgb -> palette 1 rgb")
    y, x, batch = image_dithered.shape[:3]
    is_clamp = True

    # loop over each row, from first to second last
    for y_ in range(y - 1):
        row = image_dithered[y_].astype(np.float32)  # [x batch 3]
        next_row = np.zeros_like(row)  # [x batch 3]

        # deal with the first pixel in the row
        old_color = row[0]  # [batch 3]
        color_diffs = ((palette_sq - old_color) ** 2).sum(axis=-1)  # [palette batch]
        color = palette[color_diffs.argmin(axis=0)]  # [batch 3]
        color_diff = old_color - color  # [batch 3]
        row[0] = color
        row[1] += (7 / 16) * color_diff
        next_row[[0, 1]] += einops.einsum(BC, color_diff, "two, batch rgb -> two batch rgb")

        # loop over each pixel in the row, from second to second last
        for x_ in range(1, x - 1):
            old_color = row[x_]  # [batch 3]
            color_diffs = ((palette_sq - old_color) ** 2).sum(axis=-1)  # [colors batch]
            color = palette[color_diffs.argmin(axis=0)]
            color_diff = old_color - color
            row[x_] = color
            row[x_ + 1] += (7 / 16) * color_diff
            next_row[[x_ - 1, x_, x_ + 1]] += einops.einsum(ABC, color_diff, "three, batch rgb -> three batch rgb")

        # deal with the last pixel in the row
        old_color = row[-1]
        color_diffs = ((palette_sq - old_color) ** 2).sum(axis=-1)
        color = palette[color_diffs.argmin(axis=0)]
        color_diff = old_color - color
        row[-1] = color
        next_row[[-2, -1]] += einops.einsum(AB, color_diff, "two, batch rgb -> two batch rgb")

        # update the rows, i.e. changing current row and propagating errors to next row
        image_dithered[y_] = np.clip(row, 0, 255)
        image_dithered[y_ + 1] += next_row
        if is_clamp:
            image_dithered[y_ + 1] = np.clip(image_dithered[y_ + 1], 0, 255)

    # deal with the last row
    row = image_dithered[-1]
    for x_ in range(x - 1):
        old_color = row[x_]
        color_diffs = ((palette_sq - old_color) ** 2).sum(axis=-1)
        color = palette[color_diffs.argmin(axis=0)]
        color_diff = old_color - color
        row[x_] = color
        row[x_ + 1] += color_diff

    # deal with the last pixel in the last row
    old_color = row[-1]
    color_diffs = ((palette_sq - old_color) ** 2).sum(axis=-1)
    color = palette[color_diffs.argmin(axis=0)]
    row[-1] = color
    if is_clamp:
        row = np.clip(row, 0, 255)
    image_dithered[-1] = row
    # pbar.close()

    return image_dithered.astype(np.int32)


def _get_min_max_coords(coords: dict[Any, Float[Arr, "2 n_pixels"]]) -> tuple[float, float, float, float]:
    min_x = min(coords[:, 0].min() for coords in coords.values())
    min_y = min(coords[:, 1].min() for coords in coords.values())
    max_x = max(coords[:, 0].max() for coords in coords.values())
    max_y = max(coords[:, 1].max() for coords in coords.values())
    return min_x, min_y, max_x, max_y


def make_gcode(
    all_coords: dict[tuple[int, int, int], Int[Arr, "n_coords 2"]],
    bounding_box: tuple[tuple[float, float], tuple[float, float]],
    padding: float = 0.0,
    tiling: tuple[int, int] = (1, 1),
    speed: int = 10_000,
    end_coords: tuple[float, float] | None = None,
    plot_gcode: bool = False,
    rotate: bool = False,
) -> dict[tuple[int, int, int], list[str]]:
    """
    Generates G-code for multiple different copies of the image.
    """
    gcode_all = defaultdict(list)
    times_all = defaultdict(list)
    normalized_coords_all = defaultdict(list)
    first_last_moves_all = defaultdict(list)

    if rotate:
        for color, coords in all_coords.items():
            all_coords[color] = np.array([coords[:, 1], -coords[:, 0]]).T

    (x0, y0), (x1, y1) = bounding_box

    for x_iter in range(tiling[0]):
        for y_iter in range(tiling[1]):
            print(f"Computing tile {x_iter}, {y_iter}")
            _x0 = x0 + (x1 - x0) * x_iter / tiling[0]
            _y0 = y0 + (y1 - y0) * y_iter / tiling[1]
            _x1 = x0 + (x1 - x0) * (x_iter + 1) / tiling[0]
            _y1 = y0 + (y1 - y0) * (y_iter + 1) / tiling[1]
            _bounding_box = ((_x0 + padding, _y0 + padding), (_x1 - padding, _y1 - padding))

            gcode, times, normalized_coords = make_gcode_single(
                all_coords,
                bounding_box=_bounding_box,
                speed=speed,
                end_coords=end_coords,
            )
            for k, v in gcode.items():
                gcode_all[k].extend(v)
                first_last_moves_all[k].append((v[0], v[-2]))
            for k, v in times.items():
                times_all[k].append(v)
            for k, v in normalized_coords.items():
                normalized_coords_all[k].append(v.tolist())

    print()
    for color, times in times_all.items():
        print(f"Color {color}...")
        print(f"  ...time = sum({', '.join(f'{t:.2f}' for t in times)}) = {sum(times):.2f} minutes")
        print("  ...first/last moves:")
        for first_move, last_move in first_last_moves_all[color]:
            print(f"    {first_move} ... {last_move}")

    if plot_gcode:
        output_size = 1000
        ((x0, y0), (x1, y1)) = bounding_box
        sf = output_size / (x1 - x0)
        for color, lines in normalized_coords_all.items():
            lines = np.concatenate(lines, axis=0)
            canvas = Image.new(
                "RGB",
                (output_size, int(output_size * (y1 - y0) / (x1 - x0))),
                (255, 255, 255),
            )
            draw = ImageDraw.Draw(canvas)
            points = list(zip(sf * (lines[:, 0] - x0), sf * (lines[:, 1] - y0)))
            draw.line(points, fill=color if color != "bounding_box" else "black", width=1)
            display(canvas)

    return gcode_all


def make_gcode_single(
    all_coords: dict[tuple[int, int, int], Int[Arr, "n_coords 2"]],
    bounding_box: tuple[tuple[float, float], tuple[float, float]],
    speed: int = 10_000,
    end_coords: tuple[float, float] | None = None,
) -> dict[tuple[int, int, int], list[str]]:
    eps = 1e-3

    (x0, y0), (x1, y1) = bounding_box

    min_x, min_y, max_x, max_y = _get_min_max_coords(all_coords)

    # Figure out what the max amount is we can scale, while still fitting in bounding box
    sf = min((x1 - x0) / (max_x - min_x), (y1 - y0) / (max_y - min_y))

    # Scale coords to bounding box range
    all_coords_normalized = {}
    for color, coords in all_coords.items():
        # Normalize coords to fit bounding box
        coords[:, 0] = x0 + (coords[:, 0] - min_x) * sf
        coords[:, 1] = y0 + (coords[:, 1] - min_y) * sf
        assert coords[:, 0].min() >= x0 - eps and coords[:, 0].max() <= x1 + eps
        assert coords[:, 1].min() >= y0 - eps and coords[:, 1].max() <= y1 + eps
        all_coords_normalized[color] = coords

    min_x, min_y, max_x, max_y = _get_min_max_coords(all_coords_normalized)
    all_coords_normalized["bounding_box"] = np.array(
        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]]
    )
    print(f"  Bounding box: [{min_x:.3f}-{max_x:.3f}, {min_y:.3f}-{max_y:.3f}]")

    lines = {}
    times = {}

    lines["bounding_box"] = ["M3S250 ; raise"]
    for x, y in [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]:
        lines["bounding_box"].append(f"G1 X{x:.3f} Y{y:.3f} F5000")

    for color, coords_list in all_coords_normalized.items():
        lines[color] = ["M3S250 ; raise at the start"]

        # Add all the drawing coordinates
        for i, (x, y) in enumerate(coords_list):
            lines[color].append(f"G1 X{x:.3f} Y{y:.3f} F{speed if i > 0 else 5000}")
            if i == 0:
                lines[color].append("M3S0 ; lower (to start drawing)")

        # End the drawing
        # TODO - end in a different way, so we raise the pen then move out of the way (not sure how to deal with pen maybe scratching)
        lines[color].append("M3S250 ; raise (end drawing)")
        if end_coords is not None:
            lines[color].append(f"G1 X{end_coords[0]:.3f} Y{end_coords[1]:.3f} ; end position")

        # Print total time this will take
        diffs = all_coords_normalized[color][1:] - all_coords_normalized[color][:-1]
        distances = (diffs**2).sum(-1) ** 0.5
        distance_for_one_minute = 2025
        times[color] = distances.sum() / distance_for_one_minute

    # print("\n".join(lines[:20]))
    # pyperclip.copy("\n".join(lines[50_000:]))
    return lines, times, all_coords_normalized


# ! Demo code (ignore everything below this line)


# ! Archived code for shapes

# if self.character_path is not None:
#     img = Image.open(self.char_path).convert("L")
#     img = img.resize((size, size))
#     img_array = np.array(img)
#     black_pixels = img_array < 245  # Threshold for "black" ?
#     y_coords, x_coords = np.where(black_pixels)
#     coords = np.stack([y_coords, x_coords])
#     coords_list.append(coords)
# else:
#     t = self.shape_output_thickness if output else self.shape_thickness
#     if self.shape_type == "circle":
#         img = Image.new("L", (size * 2, size * 2), 255)
#         draw = ImageDraw.Draw(img)
#         draw.ellipse([t, t, size * 2 - t, size * 2 - t], outline=0, width=t)
#     elif self.shape_type == "rect":
#         img = Image.new("L", (size, size), 255)
#         draw = ImageDraw.Draw(img)
#         draw.rectangle([t, t, size - t, size - t], outline=0, width=t)
#     elif self.shape_type == "tri":
#         img = Image.new("L", (size, size), 255)
#         draw = ImageDraw.Draw(img)
#         h = int(size * np.sqrt(3) / 2)
#         points = [(0, size - 1), (size - 1, size - 1), (size // 2, size - h)]
#         draw.polygon(points, outline=0, width=t)
#     elif self.shape_type == "hex":
#         img = Image.new("L", (size, size), 255)
#         draw = ImageDraw.Draw(img)
#         h = int(size * np.sqrt(3) / 4)
#         points = [
#             (0, size // 2),
#             (size // 4, size - 1),
#             (3 * size // 4, size - 1),
#             (size - 1, size // 2),
#             (3 * size // 4, size // 2 - h),
#             (size // 4, size // 2 - h),
#         ]
#         draw.polygon(points, outline=0, width=t)
#     elif self.shape_type == "arc":
#         raise NotImplementedError("Not implemented these shapes yet.")

#     img_array = np.array(img)
#     black_pixels = img_array < 128
#     y_coords, x_coords = np.where(black_pixels)
#     coords = np.stack([y_coords, x_coords])
#     coords_list.append(coords)
