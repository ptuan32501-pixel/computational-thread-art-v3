import enum
import pprint
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tqdm
from jaxtyping import Float, Int
from PIL import Image, ImageDraw

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
    ) -> list[tuple[dict, Float[Arr, "2 n_pixels"], Float[Arr, "2"]]]:
        """Generates `n_shapes` random shapes, and returns a list of their coords.

        Args:
            n_shapes: Number of shapes to generate
            start_dir: Direction to start the shape in
            start_coords: Coordinates to start the shape at
            canvas_y: Height of the canvas
            canvas_x: Width of the canvas

        Returns:
            List of tuples, each containing the parameters and coordinates of a shape (and end dir).
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
            coords, _, end_dir = self.draw_curve(start_coords, start_dir, **params)

            # Crop parts of `coords` which go off the edge, and only keep ones which are in bounds
            coords = mask_coords(coords, canvas_y, canvas_x, remove=True)
            if coords.shape[-1] > 0:
                coords_list.append((params, coords, end_dir))

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


@dataclass
class Drawing:
    image_path: str
    weight_image_path: str | None
    x: int
    output_x: int

    n_shapes: int
    n_random: int
    darkness: float
    negative_penalty: float

    shape: Shape

    def __post_init__(self):
        self.image = Image.open(self.image_path).convert("L")
        self.weight_image = None if self.weight_image_path is None else Image.open(self.weight_image_path).convert("L")
        width, height = self.image.size
        self.y = int(self.x * height / width)
        self.output_sf = self.output_x / self.x
        self.output_y = int(self.y * self.output_sf)

    def create_img(self) -> tuple[list[dict], Image.Image, Int[Arr, "y x"], Float[Arr, "n_pixels 2"]]:
        # Create a copy of the target image, to draw on
        image = 1.0 - np.asarray(self.image.resize((self.x, self.y))).astype(np.float32) / 255
        weight_image = (
            None
            if self.weight_image is None
            else np.asarray(self.weight_image.resize((self.x, self.y))).astype(np.float32) / 255
        )

        # Get our starting position & direction (posn is random, direction is pointing inwards)
        start_coords = (0.1 + 0.8 * np.random.rand(2)) * np.array([self.y, self.x])
        start_coords_offset = start_coords - np.array([self.y, self.x]) / 2
        start_dir = -start_coords_offset / (np.linalg.norm(start_coords_offset) + 1e-6)

        params_list = []
        pbar = tqdm.tqdm(range(self.n_shapes))

        all_coords = []

        for step in pbar:
            # Get our random parameterized shapes
            coords_list = self.shape.get_drawing_coords_list(
                n_shapes=self.n_random,
                start_dir=start_dir,
                start_coords=start_coords,
                canvas_y=self.y,
                canvas_x=self.x,
            )

            # Turn them into integer pixels, and concat them
            pixels = [coords.astype(np.int32) for _, coords, _ in coords_list]
            n_pixels = [coords.shape[-1] for _, coords, _ in coords_list]
            pixels = np.stack([pad_to_length(p, max(n_pixels)) for p in pixels])  # (n_rand, 2, n_pix)

            # Get the pixels values of the target image at these coords
            pixel_values = image[pixels[:, 0], pixels[:, 1]]  # (n_rand, n_pix)
            pixel_values_mask = np.any(pixels != 0, axis=1)  # (n_rand, n_pix)

            # Apply negative penalty and weighting
            if self.negative_penalty > 0.0:
                # pixel_values[pixel_values < 0.0] *= 1 + self.negative_penalty
                pixel_values -= self.negative_penalty * np.maximum(0.0, self.darkness - pixel_values)

            if weight_image is not None:
                pixel_weights = weight_image[pixels[:, 0], pixels[:, 1]]  # (n_rand, n_pix)
                pixel_values_mask = pixel_values_mask.astype(pixel_values.dtype) * pixel_weights

            # Average over each pixel array
            pixel_values = (pixel_values * pixel_values_mask).sum(-1) / (pixel_values_mask.sum(-1) + 1e-8)

            # Pick the darkest shape to draw
            best_idx = np.argmax(pixel_values)
            best_params, best_coords, best_end_dir = coords_list[best_idx]

            # Subtract it from the target image, and write it to the canvas
            best_pixels = best_coords.astype(np.int32)
            image[best_pixels[0], best_pixels[1]] -= self.darkness
            params_list.append(best_params)
            all_coords.append(best_coords)

            # best_pixels_large = (best_coords * self.output_x / self.x).astype(np.int32)
            # canvas[*best_pixels_large] = 0

            # This end dir is the new start dir (same for position)
            start_dir = best_end_dir
            start_coords = best_coords[:, -1]

        # Create canvas and draw on it
        canvas = Image.new("L", (self.output_x, self.output_y), 255)
        draw = ImageDraw.Draw(canvas)
        all_coords = (self.output_sf * np.concatenate(all_coords, axis=1)).T.tolist()  # shape (n_pixels, 2)
        for (y0, x0), (y1, x1) in zip(all_coords[:-1], all_coords[1:]):
            draw.line([(x0, y0), (x1, y1)], fill="black", width=1)

        return params_list, canvas, image, np.array(all_coords)


def mask_coords(
    coords: Float[Arr, "2 n_pixels"], max_y: int, max_x: int, remove: bool = False
) -> Float[Arr, "2 n_pixels"]:
    """Masks coordinates that go out of bounds."""
    assert coords.shape[0] == 2, "Coords should have shape (2, n_pixels)"
    out_of_bounds = (coords[0] < 0) | (coords[0] >= max_y) | (coords[1] < 0) | (coords[1] >= max_x)
    return coords[:, ~out_of_bounds] if remove else np.where(out_of_bounds, 0.0, coords)


def pad_to_length(arr: np.ndarray, length: int, axis: int = -1, fill_value: float = 0):
    target_shape = list(arr.shape)
    assert length >= target_shape[axis]
    target_shape[axis] = length - target_shape[axis]
    return np.concat([arr, np.full_like(arr, fill_value=fill_value, shape=tuple(target_shape))], axis=axis)


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
