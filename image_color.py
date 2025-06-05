"""
Includes all functions related to producing full-colour images (like the stag).
"""

import copy
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import NO
from typing import Generator, Literal

import einops
import numpy as np
import plotly.express as px
import torch as t
from IPython.display import HTML, display
from jaxtyping import Float
from PIL import Image, ImageFilter
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

from coordinates import build_through_pixels_dict, pair_to_index
from misc import (
    get_color_hash,
    get_img_hash,
    get_size_mb,
    global_random_seed,
    mask_ellipse,
    palette_to_html,
)

t.classes.__path__ = []
ROOT_PATH = Path(__file__).parent
assert (ROOT_PATH / "images").exists(), f"{ROOT_PATH / 'images'} folder not found in the same directory as this file."


@dataclass
class ThreadArtColorParams:
    name: str
    x: int
    n_nodes: int | tuple
    filename: str | None  # if this is None, you need to supply `image`
    w_filename: str | None
    palette: list[tuple[int, int, int]]
    n_lines_per_color: list[int]
    n_random_lines: int | Literal["all"]
    darkness: float | list[float]
    blur_rad: int
    group_orders: str | int
    mono_filenames: dict[str, str] = field(default_factory=dict)
    palette_restriction: dict = field(default_factory=dict)
    d_coords: dict = field(default_factory=dict)
    # d_pixels: dict = field(default_factory=dict) # Replaced with `t_pixels`
    d_joined: dict = field(default_factory=dict)
    d_sides: dict = field(default_factory=dict)
    t_pixels: Tensor = field(default_factory=lambda: Tensor())
    n_consecutive: int = 0
    shape: str = "Rectangle"
    seed: int = 0
    pixels_per_batch: int = 32
    num_overlap_rows: int = 6
    other_colors_weighting: list[list[float]] = field(default_factory=list)
    # ^ can be e.g. {"white": 0.1, "*": 0.2} to give all other colors 0.2 weighting but white 0.1
    step_size: float = 1.0
    # ^ if more than 1.0 then we take slightly larger jumps over pixels when drawing lines (t_pixels uses less memory)
    debug_through_pixels_dict: bool = False
    # ^ if True, we leave all the debug print statements e.g. tensor sizes, if false then we only print post init time
    mode: Literal["color", "monochrome"] = "color"
    # ^ For black and white images (hacky)
    critical_fracs: tuple[float, float | None] = (0.02, None)
    critical_frac_penalty_power_decay: float | None = None
    # ^ So we don't start and end lines too close to each other, and in a way the gantry finds hard to deal with. The
    # critical_fracs are a tuple: first element is strict (for the kind of lines which cross over themselves) and the
    # second is lenient (for lines which don't self-cross, but are just sharp angles).
    width_to_gap_ratio: float = 1.0
    # ^ Ratio of hook width to distance between hooks, so I can see what physical piece looks like
    neg_penalty_multiplier: float = 0.0
    # ^ By default a line's score is just the average pixel value, but if this parameter is >0 then we add this many
    # extra multiples of the new negative pixel values to the score. For example if this was 1.5 and our pixel values
    # were [0.5, 0.3, 0.1] with darkness 0.2, then the scores would be [0.5, 0.3, 0.1 + 1.5 * -0.1 = -0.05], the latter
    # because subtracting 0.2 from the 3rd pixel would push it into negative values.
    flip_hook_parity: bool = True
    # ^ If True, then we leave in a different way than we arrived: this is used for the pieces made with thread, not the
    # drawn pieces.
    image: Image.Image | None = None
    # ^ for streamlit page, passing through uploaded image not filename

    @classmethod
    def from_dict(cls, args_dict: dict) -> "ThreadArtColorParams":
        return cls(**args_dict)

    def __post_init__(self):
        t0 = time.time()

        # Load in real image, also gets us the width and height
        if self.image is None:
            self.image = Image.open(str(ROOT_PATH / "images" / self.filename))
        self.y = int(self.x * (self.image.height / self.image.width))

        if self.shape.lower() in ["circle", "ellipse", "round"]:
            self.shape = "Ellipse"
        else:
            self.shape = "Rectangle"

        if self.critical_fracs[1] is None:
            self.critical_fracs = (self.critical_fracs[0], self.critical_fracs[0])
        assert self.critical_fracs[0] >= self.critical_fracs[1], (
            "[0] should be bigger i.e. more restrictive (it's for the self-crossing lines), [1] should be smaller"
        )

        self.d_coords, self.d_joined, self.d_sides, self.t_pixels = build_through_pixels_dict(
            self.x,
            self.y,
            self.n_nodes,
            shape=self.shape,
            critical_fracs=self.critical_fracs,
            only_return_d_coords=False,
            width_to_gap_ratio=self.width_to_gap_ratio,
            step_size=self.step_size,
            debug=self.debug_through_pixels_dict,
        )
        print(f"ThreadArtColorParams.__init__ done in {time.time() - t0:.2f} seconds")

    @property
    def group_orders_list(self) -> list[int]:
        # Group orders is either 4 or "4" (indicating 4 copies of the colors in order), or something like "1,2,3,4,3,4"
        # indicating we do all of the 1st color, then all of the 2nd, then 50% of the 3rd, etc. We only work with the
        # `group_orders_list` object, since that's easier.
        self.group_orders = str(self.group_orders)
        if self.group_orders.isdigit():
            group_orders_list = list(range(len(self.palette))) * int(self.group_orders)
        else:
            assert all(x.isdigit() for x in self.group_orders.split(",")), (
                f"Invalid group orders: {self.group_orders}. Must be a number or comma-separated list of numbers."
            )
            group_orders_list = [int(i) - 1 for i in self.group_orders.split(",")]
        return group_orders_list

    def __repr__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                print(f"{k:>22} : tensor of shape {tuple(v.shape)}")
            elif isinstance(v, dict):
                print(f"{k:>22} : dict of length {len(v)}")
            elif k == "palette":
                s = f"<code>{'&nbsp;' * 13}palette : </code>" + palette_to_html(v.keys())
                display(HTML(s))
            elif k == "group_orders":
                raise NotImplementedError()
            elif not (k.startswith("color")):
                print(f"{k:>22} : {v}")
        return ""


# ===================================================================================================


# Class for images: contains Floyd-Steinberg dithering image function, histogram of colours, different versions of the image, etc
class Img:
    def __init__(
        self,
        args: ThreadArtColorParams,
    ) -> None:
        # TODO - maybe reuse these parameters
        self.dithering_params = ["clamp"]
        self.pixels_per_batch = 32
        self.num_overlap_rows = 6

        t0 = time.time()

        self.args = args
        self.filename = ROOT_PATH / f"images/{args.filename}" if args.filename else None
        self.save_dir = ROOT_PATH / "outputs" / args.name
        self.x = args.x
        self.y = args.y

        # Get base image (and also image converted to monochrome)
        base_image = (args.image or Image.open(self.filename)).resize((self.x, self.y))
        self.imageRGB = t.tensor((base_image).convert(mode="RGB").getdata()).reshape((self.y, self.x, 3))
        self.imageBW = t.tensor((base_image).convert(mode="L").getdata()).reshape((self.y, self.x))

        # Get the monochrome images (i.e. the ones we're using for specific colors)
        if args.mono_filenames:
            raise NotImplementedError()
            # self.imagesMono = {}
            # for color_name, filename in args.mono_filenames.items():
            #     base_image_mono = Image.open(ROOT_PATH / f"images/{filename}").resize((self.x, self.y))
            #     self.imagesMono[color_name] = (
            #         t.tensor((base_image_mono).convert(mode="L").getdata()).reshape((self.y, self.x)) / 255
            #     )

        # Process the importance weighting (we'll apply this to all images)
        self.w = None
        if args.w_filename:
            self.w_filename = ROOT_PATH / f"images/{args.w_filename}"
            base_image_w = Image.open(self.w_filename).resize((self.x, self.y))
            self.w = 1 - (t.tensor((base_image_w).convert(mode="L").getdata()).reshape((self.y, self.x)) / 255)

        # if args.wneg_filename:
        #     self.wneg_filename = "images/{}".format(args.wneg_filename)
        #     base_image_wneg = Image.open(self.wneg_filename).resize((self.x, self.y))
        #     self.wneg = 1 - (t.tensor((base_image_wneg).convert(mode="L").getdata()).reshape((self.y, self.x)) / 255)

        self.w_restriction = None

        if args.palette_restriction:
            raise NotImplementedError()
            # self.w_restriction_filename = ROOT_PATH / f"images/{args.palette_restriction['filename']}"
            # base_image_w = Image.open(self.w_restriction_filename).resize((self.x, self.y))
            # self._w_restriction = 1 - (
            #     t.tensor((base_image_w).convert(mode="L").getdata()).reshape((self.y, self.x)) / 255
            # )
            # # Convert self.w_restriction into shape (palette, y, x) with 1 in the illegal colors & positions
            # self.w_restriction = t.zeros((self.y, self.x, len(args.palette)))
            # for (lower, upper), filter_fn in args.palette_restriction["filters"].items():
            #     mask_to_apply_filter = (self._w_restriction >= lower) & (self._w_restriction <= upper)
            #     for j, color_name in enumerate(self.palette.keys()):
            #         if not filter_fn(color_name):
            #             self.w_restriction[..., j][mask_to_apply_filter] = 1.0

        if self.args.mode == "color":
            # ! Dither the images
            # This does dithering by batch, i.e. it arranges the 3D image into 4D and vectorizes the operation
            self.image_dithered, t_FS = self.FS_dither(self.pixels_per_batch, self.num_overlap_rows)

            # ! Create the mono images dict
            # Note, most colors will come from dithering `self.imageRGB`, but we also support some mono images coming from
            # images specifically provided for this purpose: `self.imagesMono`
            self.generate_mono_images_dict(self.args.other_colors_weighting)

        else:
            # Basic operation for monochrome - only black and white
            self.generate_mono_images_dict()
            t_FS = 0.0

        print(f"Other init operations complete in {time.time() - t0 - t_FS:.2f} seconds")

    # Performs FS-dithering with progress bar, returns the output (called in __init__)
    def FS_dither(
        self,
        pixels_per_batch: int | None,
        num_overlap_rows: int,
    ) -> tuple[Float[Tensor, "y x 3"], float]:
        t_FS = time.time()

        image_dithered: Float[Tensor, "y x 3"] = self.imageRGB.clone().float()
        y, x = image_dithered.shape[:2]

        image_palette_restriction = None

        if pixels_per_batch is None:
            if self.w_restriction is not None:
                image_palette_restriction = self.w_restriction.unsqueeze(-2)
            return self.FS_dither_batch(image_dithered.unsqueeze(-2), image_palette_restriction)

        num_batches = math.ceil(y / pixels_per_batch)
        rows_to_extend_by = num_batches - (y % num_batches)

        # Add a batch dimension
        image_dithered = einops.rearrange(
            t.concat([image_dithered, t.zeros(rows_to_extend_by, self.x, 3)]),
            "(batch y) x rgb -> y x batch rgb",
            batch=num_batches,
        )
        # Concat the last `num_overlap_rows` to the start of the image
        end_of_each_batch = t.concat(
            [
                t.zeros(num_overlap_rows, x, 1, 3),
                image_dithered[-num_overlap_rows:, :, :-1],
            ],
            dim=-2,
        )
        image_dithered = t.concat([end_of_each_batch, image_dithered], dim=0)

        # # Plot slice images, to check that this is working correctly
        # for batch_no in range(image_dithered.size(2)):
        #     i = image_dithered[:, :, batch_no, :]
        #     px.imshow(i).show()

        # If we're using a palette restriction in certain regions, define that
        if self.w_restriction is not None:
            image_palette_restriction = einops.rearrange(
                t.concat(
                    [
                        self.w_restriction,
                        t.zeros(rows_to_extend_by, self.x, len(self.args.palette)),
                    ]
                ),
                "(batch y) x palette -> y x palette batch",
                batch=num_batches,
            )
            end_of_each_batch = t.concat(
                [
                    t.zeros(num_overlap_rows, x, len(self.args.palette), 1),
                    image_palette_restriction[-num_overlap_rows:, :, :, :-1],
                ],
                dim=-1,
            )
            image_palette_restriction = t.concat([end_of_each_batch, image_palette_restriction], dim=0)

        image_dithered, t_FS_inner = self.FS_dither_batch(image_dithered, image_palette_restriction)

        image_dithered = einops.rearrange(
            image_dithered[num_overlap_rows - 1 : -1],
            "y x batch rgb -> (batch y) x rgb",
        )[1 : y + 1]

        t_FS = time.time() - t_FS
        print(f"FS dithering complete in {t_FS:.2f}s")

        return image_dithered, t_FS

    def FS_dither_batch(
        self,
        image_dithered: Float[Tensor, "y x batch 3"],
        image_palette_restriction: Float[Tensor, "y x palette batch"] | None,
    ) -> tuple[Tensor, float]:
        # Define the constants we'll multiply with when "shifting the errors" in dithering
        AB = t.tensor([3, 5]) / 16
        ABC = t.tensor([3, 5, 1]) / 16
        BC = t.tensor([5, 1]) / 16

        # # We don't use the colors in `imagesMono` for dithering; they have their own mono images we get the lines from
        # palette_subset = {name: color for name, color in self.palette.items() if name not in self.imagesMono}
        # palette = t.tensor(list(palette_subset.values())).float()  # [palette 3]
        palette = t.tensor(self.args.palette).float()  # [palette 3]

        # Set up stuff
        palette_sq = einops.rearrange(palette, "palette rgb -> palette 1 rgb")
        t0 = time.time()
        y, x, batch = image_dithered.shape[:3]
        is_clamp = "clamp" in self.dithering_params

        # loop over each row, from first to second last
        pbar = tqdm(range(y - 1), desc="Floyd-Steinberg dithering")
        for y_ in pbar:
            row = image_dithered[y_].float()  # [x batch 3]
            next_row = t.zeros_like(row)  # [x batch 3]

            # deal with the first pixel in the row
            old_color = row[0]  # [batch 3]
            color_diffs = (palette_sq - old_color).pow(2).sum(dim=-1)  # [palette batch]
            color = palette[color_diffs.argmin(dim=0)]  # [batch 3]
            color_diff = old_color - color  # [batch 3]
            row[0] = color
            row[1] += (7 / 16) * color_diff
            next_row[[0, 1]] += einops.einsum(BC, color_diff, "two, batch rgb -> two batch rgb")

            # loop over each pixel in the row, from second to second last
            for x_ in range(1, self.x - 1):
                old_color = row[x_]  # [batch 3]
                color_diffs = (palette_sq - old_color).pow(2).sum(dim=-1)  # [colors batch]
                if (
                    image_palette_restriction is not None
                ):  # restrict palette by adding large number to diffs of certain colors
                    color_diffs += image_palette_restriction[y_, x_] * 1e6
                color = palette[color_diffs.argmin(dim=0)]
                color_diff = old_color - color
                row[x_] = color
                row[x_ + 1] += (7 / 16) * color_diff
                next_row[[x_ - 1, x_, x_ + 1]] += einops.einsum(ABC, color_diff, "three, batch rgb -> three batch rgb")

            # deal with the last pixel in the row
            old_color = row[-1]
            color_diffs = (palette_sq - old_color).pow(2).sum(dim=-1)
            color = palette[color_diffs.argmin(dim=0)]
            color_diff = old_color - color
            row[-1] = color
            next_row[[-2, -1]] += einops.einsum(AB, color_diff, "two, batch rgb -> two batch rgb")

            # update the rows, i.e. changing current row and propagating errors to next row
            image_dithered[y_] = t.clamp(row, 0, 255)
            image_dithered[y_ + 1] += next_row
            if is_clamp:
                image_dithered[y_ + 1] = t.clamp(image_dithered[y_ + 1], 0, 255)

        # deal with the last row
        row = image_dithered[-1]
        for x_ in range(self.x - 1):
            old_color = row[x_]
            color_diffs = (palette_sq - old_color).pow(2).sum(dim=-1)
            color = palette[color_diffs.argmin(dim=0)]
            color_diff = old_color - color
            row[x_] = color
            row[x_ + 1] += color_diff

        # deal with the last pixel in the last row
        old_color = row[-1]
        color_diffs = (palette_sq - old_color).pow(2).sum(dim=-1)
        color = palette[color_diffs.argmin(dim=0)]
        row[-1] = color
        if is_clamp:
            row = t.clamp(row, 0, 255)
        image_dithered[-1] = row

        pbar.close()

        return image_dithered.to(t.int), time.time() - t0

    # Displays image output
    def display_output(self, height: int, width: int):
        if self.args.mode == "color":
            image_dithered = (
                mask_ellipse(self.image_dithered.float() / 255, 0.5)
                if self.args.shape == "Ellipse"
                else self.image_dithered.float()
            )
            px.imshow(image_dithered, height=height, width=width, template="plotly_dark").show()
        mono_images = [
            (mask_ellipse(x, 0.5) if self.args.shape == "Ellipse" else x) for x in self.mono_images_dict.values()
        ]
        fig = px.imshow(
            t.stack(mono_images),
            height=height + 120,
            width=width,
            template="plotly_dark",
            title="Images per color (white = 1, black = 0)",
            color_continuous_scale="gray",
            animation_frame=0,
        ).update_layout(coloraxis_showscale=False)
        fig.layout.sliders[0].currentvalue.prefix = "color = "
        for i, color_tuple in enumerate(self.args.palette):
            fig.layout.sliders[0].steps[i].label = str(color_tuple)
        fig.show()

    # Takes FS output and returns a dictionary of monochromatic images (called in __init__)
    def generate_mono_images_dict(self, other_colors_weighting: list[list[float]] | None = None):
        if other_colors_weighting is None:
            # Assumed monochrome mode

            imageBW_01 = self.imageBW.float() / 255.0

            self.color_histogram = {
                (0, 0, 0): 1 - imageBW_01.mean(),
                (255, 255, 255): imageBW_01.mean(),
            }
            self.mono_images_dict = {
                (0, 0, 0): 1 - imageBW_01,
                (255, 255, 255): imageBW_01,
            }

        else:
            d_histogram = dict()  # histogram of frequency of colors (cropped to a circle if necessary)
            d_mono_images_pre = dict()  # mono-color images, before they've been processed
            d_mono_images_post = dict()  # mono-color images, after processing (i.e. adding weight to nearby colors)

            # For each color, get its boolean map in the dithered image, and calculate the histogram
            for color_tuple in self.args.palette:
                # Case 1: color doesn't have its own dedicated monoImage
                # if color_name not in self.imagesMono:
                mono_image = (get_img_hash(self.image_dithered) == get_color_hash(t.tensor(color_tuple))).int()
                d_mono_images_pre[color_tuple] = mono_image
                d_histogram[color_tuple] = mono_image.sum() / mono_image.numel()
                # else:
                #     mono_image = self.imagesMono[
                #         color_name
                #     ]  # it's already a 2D monochrome image, with black = important areas
                #     d_mono_images_pre[color_name] = mono_image
                #     d_histogram[color_name] = mono_image.sum() / mono_image.numel()

            # Renormalize d_histogram (because if we used mono images for specific colors, then they won't sum to 1)
            nTotal = sum(d_histogram.values())
            d_histogram = {color_tuple: n / nTotal for color_tuple, n in d_histogram.items()}

            # Use `other_colors_weighting`, converting the mono images into linear multiples of themselves
            if other_colors_weighting:
                assert len(other_colors_weighting) == len(self.args.palette), (
                    "Should either give full list for other_colors_weighting (with optional empty elems) or empty list"
                )
                for i, base_color in enumerate(self.args.palette):
                    d_mono_images_post[base_color] = sum(
                        d_mono_images_pre[adj_color].clone().float() * adj_coeff
                        for adj_color, adj_coeff in zip(self.args.palette, other_colors_weighting[i])
                    )
            else:
                d_mono_images_post = d_mono_images_pre

            self.color_histogram = d_histogram
            self.mono_images_dict = d_mono_images_post

    # Prints a suggested number of lines, in accordance with histogram frequencies (used in Juypter Notebook)
    def decompose_image(self, n_lines_total=10000):
        table = Table("Color", "Example", "Lines")

        n_lines_per_color = [
            int(self.color_histogram[color_tuple] * n_lines_total) for color_tuple in self.args.palette
        ]

        # If we don't have the right sum, add more lines to the darkest color
        darkest_idx = [
            i
            for i, color_tuple in enumerate(self.args.palette)
            if sum(color_tuple) == max([sum(cv) for cv in self.args.palette])
        ][0]
        n_lines_per_color[darkest_idx] += n_lines_total - sum(n_lines_per_color)

        for idx, color_tuple in enumerate(self.args.palette):
            color_tuple_str_spaced = "(" + ",".join([f"{x:>3}" for x in color_tuple]) + ")"
            color_tuple_str_stripped = color_tuple_str_spaced.replace(" ", "")
            table.add_row(
                color_tuple_str_spaced,
                f"[rgb{color_tuple_str_stripped}]████████[/]",
                str(n_lines_per_color[idx]),
            )

        rprint(table)

    # Creates the actual art
    def create_canvas(self) -> dict[str, list[tuple[int, int]]]:
        line_dict = defaultdict(list)
        for color, i, j in self.create_canvas_generator():
            line_dict[color].append((i, j))

        return line_dict

    def create_canvas_generator(self) -> Generator:
        assert len(self.args.palette) == len(self.args.n_lines_per_color), (
            "Palette and lines per color don't match. Did you change the palette without re-updating params?"
        )
        t0 = time.time()

        darkness = (
            self.args.darkness
            if isinstance(self.args.darkness, list)
            else [self.args.darkness] * len(self.args.palette)
        )

        mono_image_dict = {
            color_tuple: blur_image(mono_image, self.args.blur_rad)
            for color_tuple, mono_image in self.mono_images_dict.items()
        }

        # Setting a random seed at the start of this function ensures the lines will be the same (unless params change)
        global_random_seed(self.args.seed)

        pbar = tqdm(desc="Creating canvas", total=sum(self.args.n_lines_per_color))
        for color_idx, color_tuple in enumerate(self.args.palette):
            # Setup variables (including the place we'll start)
            n_lines = self.args.n_lines_per_color[color_idx]
            m_image = mono_image_dict[color_tuple]

            pbar.set_postfix_str(f"Current color: {color_tuple}")

            # Choose starting node (i.e. the first node to draw a line from)
            i = np.random.choice(list(self.args.d_joined.keys())).item()

            for n in range(n_lines):  # range(n_lines): #, leave=False):
                # Choose and add line
                j = self.choose_and_subtract_best_line(m_image=m_image, i=i, darkness=darkness[color_idx])
                yield color_tuple, i, j

                # Get the outgoing node
                i = (j + 1 if (j % 2 == 0) else j - 1) if self.args.flip_hook_parity else j

                # Maybe jump randomly to a non-consecutive node, for svg security
                if self.args.n_consecutive != 0 and ((n + 1) % self.args.n_consecutive) == 0:
                    i = list(self.args.d_joined.keys())[t.randint(0, len(self.args.d_joined), (1,))]

            # Update progress bar
            pbar.update(n_lines)

        # If not verbose then we don't have a progress bar, just a single printout at the end
        print(f"Created canvas in {time.time() - t0:.2f} seconds")

    # Generates a bunch of random lines and chooses the best one
    def choose_and_subtract_best_line(self, m_image: Tensor, i: int, darkness: float) -> int:
        """
        Generates a bunch of random lines (choosing them from `d_joined` which is a dictionary mapping node ints to all the
        nodes they're connected to), picks the best line, subtracts its darkness from the image, and returns that line.
        """
        w = self.w
        n_random_lines = self.args.n_random_lines
        neg_penalty_multiplier = self.args.neg_penalty_multiplier
        d_joined = self.args.d_joined
        t_pixels = self.args.t_pixels
        n_nodes = self.args.n_nodes
        critical_fracs = self.args.critical_fracs
        critical_frac_penalty_power_decay = self.args.critical_frac_penalty_power_decay

        # Choose `j` random lines (or as many as possible)
        if n_random_lines == "all" or n_random_lines > len(d_joined[i]):
            j_choices = t.tensor(d_joined[i]).long()
        else:
            j_choices = t.from_numpy(
                np.random.choice(d_joined[i], min(len(d_joined[i]), n_random_lines), replace=False)
            ).long()
        n_lines = j_choices.size(0)

        # Get the pixels in the line, and rearrange it
        coords_yx = t_pixels[pair_to_index(i, j_choices, n_nodes)].int()  # [n_lines 2 pixels]
        is_zero = (coords_yx == 0).all(dim=1)  # [n_lines pixels]
        coords_yx = einops.rearrange(coords_yx, "j yx pixels -> yx (j pixels)")  # [2 n_lines*pixels]

        # Get the pixels in the line, and reshape it back to [n_lines, pixels]
        pixel_values = m_image[coords_yx[0], coords_yx[1]]  # [n_lines*pixels]
        pixel_values = einops.rearrange(pixel_values, "(j pixels) -> j pixels", j=n_lines).masked_fill(is_zero, 0)

        # If any of our pixels are less than the darkness, and if neg_penalty_multiplier > 0, then we decrease their scores.
        # The amount they're decreased by equals the negative values they'll have after subtracting the darkness, scaled by
        # the neg_penalty_multiplier (e.g. if value is 0.2, darkness is 0.5, multiplier is 0.5, then we would decrease the
        # score by 0.5 * (0.5 - 0.2) = 0.15 to reflect how we're de-incentivising pushing into negative values).
        if neg_penalty_multiplier > 1e-6:
            pixel_values -= neg_penalty_multiplier * (darkness - pixel_values).clamp(min=0.0)

        # Optionally index & rearrange the weighting, in the same way as the pixels
        lengths = (~is_zero).sum(-1).float()  # [n_lines]

        if isinstance(w, Tensor):
            w_pixel_values = w[coords_yx[0], coords_yx[1]]
            w_pixel_values = einops.rearrange(w_pixel_values, "(j pixels) -> j pixels", j=n_lines).masked_fill(
                is_zero, 0
            )
            w_sum = w_pixel_values.sum(dim=-1)  # [n_lines]
            scores = (pixel_values * w_pixel_values).sum(-1) / w_sum  # [n_lines]
        else:
            scores = pixel_values.sum(-1) / lengths  # [n_lines]

        # Add penalties to the scores for short lines. For example, if we aren't allowing clockwise lines of length 20,
        # then we apply a probabilistic filter to lines of length between 20 and 20 * 2 = 40. This gives us a smooth
        # gradient of lines from length 20 to 30, rather than a bunch of lines at 20 creating a radial effect. Note that
        # we deal with clockwise and anticlockwise differently, because they have different thresholds.
        if critical_frac_penalty_power_decay is not None:
            assert critical_frac_penalty_power_decay > 0.0, "Power decay penalty must be in (0, 1] range"

            if self.args.shape == "Rectangle":
                raise NotImplementedError()
                # n_nodes_on_adj_side = nodes_per_side_list[(i_side + 1) % 4]
                # ac_corner = starting_idx_list[(i_side + 1) % 4]  # node at the corner anticlockwise to `i`
                # c_corner = starting_idx_list[i_side]  # node at the corner clockwise to `i`
                # assert c_corner <= i < (ac_corner or n4), f"Error: {c_corner=}, {i=}, {ac_corner=}"

                # banned_anticlockwise = range(
                #     ac_corner, ac_corner + min(n_nodes_on_adj_side, int((ac_corner - i) * critical_fracs[i % 2]))
                # )
                # banned_clockwise = range(
                #     c_corner - min(n_nodes_on_adj_side, int((i - c_corner) * critical_fracs[1 - i % 2])), c_corner
                # )
                # d_joined[i] = sorted(set(d_joined[i]) - set(banned_anticlockwise) - set(banned_clockwise))
            else:
                critical_frac_ac, critical_frac_c = critical_fracs[::-1] if i % 2 == 1 else critical_fracs
                diff_angles_ac = (j_choices - i) % n_nodes / n_nodes  # in range [critical_frac_ac, 1]
                diff_angles_c = (i - j_choices) % n_nodes / n_nodes  # in range [critical_frac_c, 1]

            assert diff_angles_ac.min() >= critical_frac_ac, f"Error: {diff_angles_ac.min()} < {critical_frac_ac}"
            assert diff_angles_c.min() >= critical_frac_c, f"Error: {diff_angles_c.min()} < {critical_frac_c}"

            # penalty_ac should be 1.0 at critical_frac_ac, and 0.0 at 2 * critical_frac_ac
            penalty_ac = t.clamp((2 * critical_frac_ac - diff_angles_ac) / critical_frac_ac, min=0.0, max=1.0)
            penalty_c = t.clamp((2 * critical_frac_c - diff_angles_c) / critical_frac_c, min=0.0, max=1.0)
            assert not ((penalty_ac > 0) & (penalty_c > 0)).any(), (
                "penalty_ac and penalty_c overlap on nonzero elements"
            )
            # Now we use these penalties to maybe replace the scores with neginf, removing those lines from consideration
            penalty = (penalty_ac + penalty_c) ** critical_frac_penalty_power_decay
            scores -= 1e4 * (t.rand(size=(n_lines,)) < penalty).float()

        # Now choose the best remaining option!
        best_j = j_choices[scores.argmax()].item()

        coords_yx = t_pixels[pair_to_index(i, best_j, n_nodes)].int()  # [yx=2 pixels]
        is_zero = coords_yx.sum(0) == 0  # [pixels]
        coords_yx = coords_yx[:, ~is_zero]
        m_image[coords_yx[0], coords_yx[1]] -= darkness

        return best_j

    # Creates images / animations from the art
    def paint_canvas(
        self,
        line_dict: list[tuple[int, int]],
        x_output: int | None = None,
        rand_perm: float = 0.0025,
        fraction: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
        background_color: tuple[int, int, int] | None = (0, 0, 0),
        inner_background_color: tuple[int, int, int] | None = None,
        show_individual_colors: bool = False,
        line_width_multiplier: float = 1.0,
        png: bool = True,
        verbose: bool = False,
        make_html: bool = True,
        html_line_width_multiplier: float = 1.0,
        html_x: int = 800,
        html_x_small: int | None = None,
        html_total_slider_steps: int = 150,
        html_rand_perm: float = 0.0025,
        html_bg_color: tuple[int, int, int] = (0, 0, 0),
        html_color_names: list[str] = [],
    ):
        """
        Takes the line_dict, and uses it to create an svg of the output, then saves it
        """
        t0 = time.time()
        if not self.save_dir.exists():
            self.save_dir.mkdir()

        # if fraction != (0, 1), it means you're not plotting all the lines, only a subset of them - useful to see how many lines are actually needed to make the image look good
        # precise syntax: fraction = (a, b) means you're plotting between the a and bth lines, e.g. (0, 0.5) means the best half
        line_dict = copy.deepcopy(line_dict)
        if isinstance(fraction, tuple):
            line_dict = {
                color: lines[int(fraction[0] * len(lines)) : int(fraction[1] * len(lines))]
                for color, lines in line_dict.items()
            }
        elif isinstance(fraction, dict):
            line_dict = {
                color: lines[
                    int(fraction.get(color, (0, 1))[0] * len(lines)) : int(fraction.get(color, (0, 1))[1] * len(lines))
                ]
                for color, lines in line_dict.items()
            }
        else:
            line_dict = {k: v for k, v in line_dict.items()}

        # Maybe make the HTML
        if make_html:
            html = self.generate_thread_art_html(
                line_dict,
                x=html_x,
                x_small=html_x_small,
                line_width=0.12 * html_line_width_multiplier,
                steps_per_slider=html_total_slider_steps,
                rand_perm=html_rand_perm,
                bg_color=html_bg_color,
                color_names=html_color_names,
            )
            with open(f"outputs/{self.args.name}/{self.args.name}.html", "w") as f:
                f.write(html)

        # # Possibly sub a color for a different one
        # color_dict = {k: v for k, v in self.palette.items()}
        # for k, v in color_substitution.items():
        #     assert k in color_dict, f"Color {k} not in palette."
        #     color_dict[k] = v

        # Get progress bar stuff
        progress_bar = tqdm_notebook(
            total=sum([len(lines) for lines in line_dict.values()]),
            desc="Painting canvas",
            disable=not verbose,
        )
        line_counter = 0

        # Get x and y values, and also the coords dict
        if x_output is None:
            x_output = self.x
            y_output = self.y
            d_coords = self.args.d_coords
        else:
            y_output = int(self.y * x_output / self.x)
            d_coords = build_through_pixels_dict(
                x_output,
                y_output,
                self.args.n_nodes,
                shape=self.args.shape,
                critical_fracs=self.args.critical_fracs,
                only_return_d_coords=True,
                width_to_gap_ratio=1,
            )
        YX = t.tensor([y_output, x_output])

        # Deal with case where img_name had a fwd slash in it
        img_name = self.args.name.split("/")[-1]

        import cairo

        with cairo.SVGSurface(str(self.save_dir / f"{img_name}.svg"), x_output, y_output) as surface:
            context = cairo.Context(surface)
            context.scale(x_output, y_output)
            context.set_line_width(0.0002 * line_width_multiplier)

            # If background color is specified, set it everywhere
            bg_color = [0.0, 0.0, 0.0, 0.0] if background_color is None else [c / 255 for c in background_color]
            context.set_source_rgba(*bg_color)
            context.paint()

            # If inner background color is specified (and we're using circles), set it inside the circle
            if self.args.shape == "Ellipse" and inner_background_color is not None:
                inner_bg_color = [c / 255 for c in inner_background_color]
                context.set_source_rgba(*inner_bg_color)  # set bg color
                context.arc(0.5, 0.5, 0.495, 0, 2 * math.pi)  # draw circle as 360-deg arc
                context.clip()  # clip to circle region we just drew
                context.paint()  # paint the background color

            for i_idx, i in enumerate(self.args.group_orders_list):
                color_tuple = self.args.palette[i]
                lines = line_dict[color_tuple]
                context.set_source_rgb(*[c / 255 for c in color_tuple])

                n_groups = len([j for j in self.args.group_orders_list if j == i])
                group_order = len([j for j in self.args.group_orders_list[:i_idx] if j == i])

                n = int(len(lines) / n_groups)
                lines_to_draw = lines[::-1][n * group_order : n * (group_order + 1)]

                current_node = -1

                for line in lines_to_draw:
                    starting_node = line[1]
                    if starting_node != current_node:
                        y, x = d_coords[starting_node] / YX
                        y, x = hacky_permutation(y, x, rand_perm)
                        context.move_to(x, y)

                    finishing_node = line[0]
                    y, x = d_coords[finishing_node] / YX
                    y, x = hacky_permutation(y, x, rand_perm)
                    context.line_to(x, y)

                    current_node = finishing_node
                    line_counter += 1
                progress_bar.update(line_counter - progress_bar.n)

                context.stroke()

            # Get a final PNG, if necessary
            if png:
                surface.write_to_png(str(self.save_dir / f"{img_name}.png"))

        if show_individual_colors:
            for color_idx, color_tuple in enumerate(self.args.palette):
                lines = line_dict[color_tuple]
                with cairo.SVGSurface(
                    str(self.save_dir / f"{img_name}_{color_idx}.svg"),
                    x_output,
                    y_output,
                ) as surface:
                    context = cairo.Context(surface)
                    context.scale(x_output, y_output)
                    context.set_line_width(0.0002 * line_width_multiplier)
                    # Set background color either black or white, whichever is more appropriate
                    use_black_background = sum(color_tuple) >= 255 * 2
                    bg_color = [0.0, 0.0, 0.0] if use_black_background else [1.0, 1.0, 1.0]
                    context.set_source_rgb(*bg_color)
                    context.paint()
                    context.set_source_rgb(*[c / 255 for c in color_tuple])

                    current_node = -1

                    for line in lines:
                        starting_node = line[1]
                        if starting_node != current_node:
                            y, x = d_coords[starting_node] / YX
                            y, x = hacky_permutation(y, x, rand_perm)
                            context.move_to(x, y)

                        finishing_node = line[0]
                        y, x = d_coords[finishing_node] / YX
                        y, x = hacky_permutation(y, x, rand_perm)
                        context.line_to(x, y)

                        current_node = finishing_node

                    context.stroke()

            # If not verbose, we don't have a progress bar, just a single time printout at the end
            if not verbose:
                print(f"Painted canvas in {time.time() - t0:.2f} seconds")

    def generate_thread_art_instructions_html(
        self,
        line_dict: dict[tuple[int, int, int], list[tuple[int, int]]],
        dark_mode: bool = True,
        next_arrow_size: float = 0.5,
    ) -> str:
        # Calculate total number of lines
        total_lines = sum(len(lines) for lines in line_dict.values())

        # Process the lines in the order specified by group_orders
        ordered_lines = []
        color_indices = []  # Track which color each line belongs to
        slice_indices = []  # Track which slice each line belongs to

        for i_idx, i in enumerate(self.args.group_orders_list):
            color_tuple = list(line_dict.keys())[i]
            lines = line_dict[color_tuple]

            # Count occurrences of this group in total and so far
            total_occurrences = self.args.group_orders_list.count(i)
            current_occurrence = self.args.group_orders_list[: i_idx + 1].count(i)

            # Calculate lines per occurrence, ensuring all lines are used
            base_lines_per_group = len(lines) // total_occurrences
            remainder = len(lines) % total_occurrences

            # Calculate start and end indices for this occurrence
            start_idx = (base_lines_per_group * (current_occurrence - 1)) + min(current_occurrence - 1, remainder)
            end_idx = (base_lines_per_group * current_occurrence) + min(current_occurrence, remainder)

            lines_to_draw = lines[::-1][start_idx:end_idx]  # Keep the reversal as requested

            for line in lines_to_draw:
                ordered_lines.append(list(line))
                color_indices.append(i)
                slice_indices.append(i_idx)

        # Calculate information needed for the fractional displays
        color_line_counts = {}  # Total lines per color
        for i, color_tuple in enumerate(line_dict.keys()):
            color_line_counts[i] = len(line_dict[color_tuple])

        # Calculate lines per slice
        slice_line_counts = {}
        for i_idx, i in enumerate(self.args.group_orders_list):
            # Same calculation as above to determine how many lines are in this slice
            total_occurrences = self.args.group_orders_list.count(i)
            current_occurrence = self.args.group_orders_list[: i_idx + 1].count(i)

            base_lines_per_group = len(line_dict[list(line_dict.keys())[i]]) // total_occurrences
            remainder = len(line_dict[list(line_dict.keys())[i]]) % total_occurrences

            lines_in_slice = base_lines_per_group + (1 if current_occurrence <= remainder else 0)
            slice_line_counts[i_idx] = lines_in_slice

        # Convert colors to strings for display
        color_strings = [str(list(line_dict.keys())[i]) for i in range(len(line_dict))]

        # Set colors based on dark_mode
        bg_color = "#000000" if dark_mode else "#FFFFFF"
        even_arrow_color = "#FFFFFF" if dark_mode else "#000000"
        odd_arrow_color = "#FF0000"  # Red for both modes
        text_color = "#FFFFFF" if dark_mode else "#000000"

        style_replace_dict = {
            "TEXT_COLOR": text_color,
            "BG_COLOR": bg_color,
        }

        return f"""
{load_template("instructions-index.html")}

<style>
{load_template("instructions-style.css", style_replace_dict)}
</style>

<script>
const keepColorName = true;
const n_nodes = {self.args.n_nodes};
const totalLines = {total_lines};
const orderedLines = {ordered_lines};
const colorIndices = {color_indices};
const sliceIndices = {slice_indices};
const colorStrings = {color_strings};
const colorLineCounts = {color_line_counts};
const sliceLineCounts = {slice_line_counts};
const evenArrowColor = "{even_arrow_color}";
const oddArrowColor = "{odd_arrow_color}";
const nextArrowSize = {next_arrow_size};

{load_template("instructions-init.js")}
</script>
"""

    def generate_thread_art_html(
        self,
        line_dict: dict[tuple[int, int], list[tuple[int, int]]],
        x: int = 800,
        x_small: int | None = None,
        line_width: float = 0.12,
        steps_per_slider: int = 150,
        rand_perm: float = 0.0025,
        bg_color: tuple[int, int, int] = (0, 0, 0),
        color_names: list[str] = [],
    ) -> str:
        group_orders_total = {i: self.args.group_orders_list.count(i) for i in set(self.args.group_orders_list)}
        group_orders_count = {i: 0 for i in set(self.args.group_orders_list)}

        # Calculate dimensions
        full_width = x
        full_height = int(full_width * self.args.y / self.args.x)

        # Individual color images will be half the scale
        x_small = x_small or int(0.4 * x)
        small_width = int(x_small)
        small_height = int(small_width * self.args.y / self.args.x)

        data = {
            "d_coords": {int(k): [round(c, 2) for c in v.tolist()] for k, v in self.args.d_coords.items()},
            "palette": [f"rgb{k}" for k in self.args.palette],
            "group_orders_list": self.args.group_orders_list,
            "group_orders_total": group_orders_total,
            "group_orders_count": group_orders_count,
            "line_dict": {f"rgb{k}": v[::-1] for k, v in line_dict.items()},
        }

        return f"""
{load_template("index.html")}

<style>
{load_template("style.css")}
</style>

<script>
const smallWidth = {small_width};
const smallHeight = {small_height};
const fullWidth = {full_width};
const fullHeight = {full_height};
const randPerm = {rand_perm};
const nSteps = {steps_per_slider}; // Total number of steps
const lineWidth = {line_width};
const bgColor = 'rgb{bg_color}';
const colorNames = {json.dumps(color_names)};

const data = {json.dumps(data)};
console.log(data);

{load_template("init.js")}
</script>
"""


def load_template(filename: str, replace_dict: dict = {}) -> str:
    path = Path(__file__).parent / "templates" / filename
    assert path.exists()
    content = path.read_text()
    for key, value in replace_dict.items():
        content = content.replace(key, str(value))
    return content


# Blurs the monochromatic images (used in the function below)
def linear_blur_image(image: Tensor, rad: int, threeD=False):
    if rad == 0:
        return image

    # define the matrix, and normalise it
    mat = t.zeros(2 * rad + 1, 2 * rad + 1)
    # std = 0.5 * rad
    for x in range(2 * rad + 1):
        x_offset = x - rad
        for y in range(2 * rad + 1):
            y_offset = y - rad
            value = 1 - (abs(y_offset) + abs(x_offset)) / (2 * rad + 1)  # 1/(abs(y_offset)+abs(x_offset)+1)
            mat[y, x] = value
    mat = mat / mat.sum()

    # create a canvas larger than the image, then loop over the matrix adding the appropriate copies of the canvas
    if threeD:
        image_size_y, image_size_x, _ = image.shape
        canvas_size_y = image.size(0) + 2 * rad
        canvas_size_x = image.size(1) + 2 * rad
        canvas = t.zeros(canvas_size_y, canvas_size_x, 3)
        for x in range(2 * rad + 1):
            for y in range(2 * rad + 1):
                canvas[y : y + image_size_y, x : x + image_size_x, :] += mat[y, x] * image
    else:
        image_size_y, image_size_x = image.shape
        canvas_size_y = image.size(0) + 2 * rad
        canvas_size_x = image.size(1) + 2 * rad
        canvas = t.zeros(canvas_size_y, canvas_size_x)
        for x in range(2 * rad + 1):
            for y in range(2 * rad + 1):
                canvas[y : y + image_size_y, x : x + image_size_x] += mat[y, x] * image

    # crop the canvas, and return it
    return canvas[rad:-rad, rad:-rad]


# Performs either linear blurring (image above) or Gaussian (used in `create_canvas` function, for image processing before creating output)
def blur_image(img: Tensor, rad: int, mode="linear", **kwargs) -> Tensor:
    if rad == 0:
        return img

    if mode == "linear":
        return linear_blur_image(img, rad, **kwargs)

    elif mode == "gaussian":
        # We need to go through torch -> numpy -> image (filter) -> numpy -> torch
        return t.from_numpy(np.asarray(Image.fromarray(img.numpy()).filter(ImageFilter.GaussianBlur(radius=rad)))).to(
            t.float
        )

    else:
        raise ValueError("Mode must be either 'linear' or 'gaussian'.")


# Permutes coordinates, to stop weird-looking line pattern effects (used by `paint_canvas` function)
def hacky_permutation(y, x, r):
    R = r * (2 * np.random.random() - 1)

    if (x < 0.01) or (x > 0.99):
        return y + R, x
    else:
        return y, x + R
