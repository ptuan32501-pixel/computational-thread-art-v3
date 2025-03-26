"""
Includes all functions related to producing full-colour images (like the stag).
"""

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Literal, Optional, Tuple

import cv2
import einops
import numpy as np
import plotly.express as px
import torch as t
from IPython.display import HTML, clear_output, display
from jaxtyping import Float
from PIL import Image, ImageFilter
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

from coordinates import build_through_pixels_dict
from misc import (
    get_color_hash,
    get_img_hash,
    global_random_seed,
    mask_ellipse,
    myprogress,
    palette_to_html,
)

ROOT_PATH = Path(__file__).parent
assert (ROOT_PATH / "images").exists(), f"{ROOT_PATH / 'images'} folder not found in the same directory as this file."


@dataclass
class ThreadArtColorParams:
    name: str
    x: int
    n_nodes: int | tuple
    filename: str
    w_filename: Optional[str]
    palette: dict[str, tuple[int, int, int]]
    n_lines_per_color: list[int]
    n_random_lines: int | Literal["all"]
    darkness: float | dict[str, float]
    blur_rad: int
    group_orders: str | int
    mono_filenames: dict[str, str] = field(default_factory=dict)
    palette_restriction: dict = field(default_factory=dict)
    d_coords: dict = field(default_factory=dict)
    d_pixels: dict = field(default_factory=dict)
    d_joined: dict = field(default_factory=dict)
    d_sides: dict = field(default_factory=dict)
    t_pixels: Tensor = field(default_factory=lambda: Tensor())
    n_consecutive: int = 0
    shape: str = "Rectangle"
    seed: int = 0
    pixels_per_batch: int = 32
    num_overlap_rows: int = 6
    other_colors_weighting: dict = field(
        default_factory=dict
    )  # can be e.g. {"white": 0.1, "*": 0.2} to give all other colors 0.2 weighting but white 0.1

    def __post_init__(self):
        self.color_names = list(self.palette.keys())
        self.color_values = list(self.palette.values())
        first_color_letters = [color[0] for color in self.color_names]
        assert len(set(first_color_letters)) == len(first_color_letters), (
            "First letter of each color name must be unique."
        )

        if isinstance(self.group_orders, int):
            self.group_orders = "".join(first_color_letters) * self.group_orders

        img_raw = Image.open(str(ROOT_PATH / "images" / self.filename))
        self.y = int(self.x * (img_raw.height / img_raw.width))

        if self.shape.lower() in ["circle", "ellipse", "round"]:
            self.shape = "Ellipse"
        else:
            self.shape = "Rectangle"

        self.d_coords, self.d_pixels, self.d_joined, self.d_sides, self.t_pixels = build_through_pixels_dict(
            self.x,
            self.y,
            self.n_nodes,
            shape=self.shape,
            critical_distance=14,
            only_return_d_coords=False,
            width_to_gap_ratio=1,
        )

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
                assert isinstance(v, str)
                color_first_letter_to_index = {color_name[0]: i for i, color_name in enumerate(self.palette.keys())}
                # TODO - fix this, it won't work
                color_list = [list(self.palette)[color_first_letter_to_index[char]] for char in v]
                s = f"<code>{'&nbsp;' * 8}group_orders : </code>" + palette_to_html(color_list)
                display(HTML(s))
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
        self.wneg = None
        self.pixels_per_batch = 32
        self.num_overlap_rows = 6

        t0 = time.time()

        self.args = args
        self.filename = ROOT_PATH / f"images/{args.filename}"
        self.save_dir = ROOT_PATH / "outputs" / args.name
        if not self.save_dir.exists():
            self.save_dir.mkdir()
        self.x = args.x
        self.y = args.y
        self.palette = {color_name: tuple(color_value) for color_name, color_value in args.palette.items()}

        # Get base image (and also image converted to monochrome)
        base_image = Image.open(self.filename).resize((self.x, self.y))
        self.imageRGB = t.tensor((base_image).convert(mode="RGB").getdata()).reshape((self.y, self.x, 3))
        self.imageBW = t.tensor((base_image).convert(mode="L").getdata()).reshape((self.y, self.x))

        # Get the monochrome images (i.e. the ones we're using for specific colors)
        self.imagesMono = {}
        for color_name, filename in args.mono_filenames.items():
            base_image_mono = Image.open(ROOT_PATH / f"images/{filename}").resize((self.x, self.y))
            self.imagesMono[color_name] = (
                t.tensor((base_image_mono).convert(mode="L").getdata()).reshape((self.y, self.x)) / 255
            )

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
            self.w_restriction_filename = ROOT_PATH / f"images/{args.palette_restriction['filename']}"
            base_image_w = Image.open(self.w_restriction_filename).resize((self.x, self.y))
            self._w_restriction = 1 - (
                t.tensor((base_image_w).convert(mode="L").getdata()).reshape((self.y, self.x)) / 255
            )
            # Convert self.w_restriction into shape (palette, y, x) with 1 in the illegal colors & positions
            self.w_restriction = t.zeros((self.y, self.x, len(args.palette)))
            for (lower, upper), filter_fn in args.palette_restriction["filters"].items():
                mask_to_apply_filter = (self._w_restriction >= lower) & (self._w_restriction <= upper)
                for j, color_name in enumerate(self.palette.keys()):
                    if not filter_fn(color_name):
                        self.w_restriction[..., j][mask_to_apply_filter] = 1.0

        # ! Dither the images
        # This does dithering by batch, i.e. it arranges the 3D image into 4D and vectorizes the operation
        self.image_dithered, t_FS = self.FS_dither(self.pixels_per_batch, self.num_overlap_rows)

        # ! Create the mono images dict
        # Note, most colors will come from dithering `self.imageRGB`, but we also support some mono images coming from
        # images specifically provided for this purpose: `self.imagesMono`
        self.generate_mono_images_dict(args.d_pixels, self.args.other_colors_weighting)

        print(f"Other init operations complete in {time.time() - t0 - t_FS:.2f} seconds")

    def save(self):
        """Saves the individual dithered images."""
        t0 = time.time()
        for color_name, mono_image in self.mono_images_dict.items():
            img = Image.fromarray(mono_image.numpy().astype(np.uint8) * 255)
            img.save(self.save_dir / f"dithered_{color_name}.png")
        img = Image.fromarray(self.image_dithered.numpy().astype(np.uint8))
        img.save(self.save_dir / "dithered.png")
        print(f"Saved images in {time.time() - t0:.2f} seconds")

    # Performs FS-dithering with progress bar, returns the output (called in __init__)
    def FS_dither(
        self,
        pixels_per_batch: int | None,
        num_overlap_rows: int,
    ) -> Tuple[Tensor, float]:
        t_FS = time.time()

        image_dithered: Float[Tensor, "y x 3"] = self.imageRGB.clone().to(t.float)
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
    ) -> Tuple[Tensor, float]:
        # Define the constants we'll multiply with when "shifting the errors" in dithering
        AB = t.tensor([3, 5]) / 16
        ABC = t.tensor([3, 5, 1]) / 16
        BC = t.tensor([5, 1]) / 16

        # We don't use the colors in `imagesMono` for dithering; they have their own mono images we get the lines from
        palette_subset = {name: color for name, color in self.palette.items() if name not in self.imagesMono}
        palette = t.tensor(list(palette_subset.values())).to(t.float)  # [palette 3]

        # Set up stuff
        palette_sq = einops.rearrange(palette, "palette rgb3 -> palette 1 rgb3")
        t0 = time.time()
        y, x, batch = image_dithered.shape[:3]
        is_clamp = "clamp" in self.dithering_params

        # loop over each row, from first to second last
        for y_ in tqdm(range(y - 1), desc="Floyd-Steinberg dithering"):
            row = image_dithered[y_].to(t.float)  # [x batch 3]
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

        clear_output()

        return image_dithered.to(t.int), time.time() - t0

    # Displays image output
    def display_output(self, height: int, width: int):
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
        fig.layout.sliders[0].currentvalue.prefix = "color = "  # type: ignore
        for i, color_name in enumerate(self.palette.keys()):
            fig.layout.sliders[0].steps[i].label = color_name  # type: ignore
        fig.show()

    # Takes FS output and returns a dictionary of monochromatic images (called in __init__)
    def generate_mono_images_dict(self, d_pixels, other_colors_weighting: dict[str, dict[str, float]]):
        # Gets the pixels which are actually relevant (i.e. just taking the ones in the d_pixels dict)
        boolean_mask = t.zeros(size=self.image_dithered.shape[:-1])
        pixels_y_all, pixels_x_all = list(zip(*d_pixels.values()))
        pixels_y_all = t.concat(pixels_y_all).long()
        pixels_x_all = t.concat(pixels_x_all).long()
        boolean_mask[pixels_y_all, pixels_x_all] = 1

        d_histogram = dict()  # histogram of frequency of colors (cropped to a circle if necessary)
        d_mono_images_pre = dict()  # mono-color images, before they've been processed
        d_mono_images_post = dict()  # mono-color images, after processing (i.e. adding weight to nearby colors)

        # For each color, get its boolean map in the dithered image, and calculate the histogram
        for color_name, color_value in self.palette.items():
            # Case 1: color doesn't have its own dedicated monoImage
            if color_name not in self.imagesMono:
                mono_image = (get_img_hash(self.image_dithered) == get_color_hash(t.tensor(color_value))).to(t.int)
                d_mono_images_pre[color_name] = mono_image
                d_histogram[color_name] = (mono_image * boolean_mask).sum() / boolean_mask.sum()
            else:
                mono_image = self.imagesMono[
                    color_name
                ]  # it's already a 2D monochrome image, with black = important areas
                d_mono_images_pre[color_name] = mono_image
                d_histogram[color_name] = (mono_image * boolean_mask).sum() / boolean_mask.sum()

        # Renormalize d_histogram (because if we used mono images for specific colors, then they won't sum to 1)
        nTotal = sum(d_histogram.values())
        d_histogram = {color_name: n / nTotal for color_name, n in d_histogram.items()}

        # Use `other_colors_weighting`, converting the mono images into linear multiples of themselves
        valid_keys = list(self.palette.keys()) + ["*"]
        for base_color_name in self.palette.keys():
            d_mono_images_post[base_color_name] = d_mono_images_pre[base_color_name].clone().to(t.float)
            if base_color_name in other_colors_weighting:
                w = other_colors_weighting[base_color_name]
                assert all(key in valid_keys for key in w), f"Invalid keys in other_colors_weighting: {w.keys()}"
                for adj_color_name in self.palette.keys():
                    if adj_color_name != base_color_name:
                        adj_color_sf = w.get(adj_color_name, w.get("*", 0.0))
                        d_mono_images_post[base_color_name] += adj_color_sf * d_mono_images_pre[adj_color_name]

        self.color_histogram = d_histogram
        self.mono_images_dict = d_mono_images_post

    # Prints a suggested number of lines, in accordance with histogram frequencies (used in Juypter Notebook)
    def decompose_image(self, n_lines_total=10000):
        table = Table("Color", "Name", "Example", "Lines")

        n_lines_per_color = [int(self.color_histogram[color] * n_lines_total) for color in self.palette]

        # If we don't have the right sum, add more lines to the darkest color
        darkest_idx = [
            i
            for i, (_, color_values) in enumerate(self.palette.items())
            if sum(color_values) == max([sum(cv) for cv in self.palette.values()])
        ][0]
        n_lines_per_color[darkest_idx] += n_lines_total - sum(n_lines_per_color)

        for idx, (color_name, color_value) in enumerate(self.palette.items()):
            color_value_str_spaced = "(" + ",".join([f"{x:>3}" for x in color_value]) + ")"
            color_value_str_stripped = color_value_str_spaced.replace(" ", "")
            table.add_row(
                color_value_str_spaced,
                color_name,
                f"[rgb{color_value_str_stripped}]████████[/]",
                str(n_lines_per_color[idx]),
            )

        rprint(table)

    # Creates the actual art
    def create_canvas(self, verbose: bool = True) -> dict[str, list[tuple[int, int]]]:
        from collections import defaultdict

        line_dict = defaultdict(list)
        for color, i, j in self.create_canvas_generator(verbose=verbose):
            line_dict[color].append((i, j))

        return line_dict

    def create_canvas_generator(self, verbose: bool = True) -> Generator:
        assert len(self.palette) == len(self.args.n_lines_per_color), (
            "Palette and lines per color don't match. Did you change the palette without re-updating params?"
        )
        t0 = time.time()
        line_dict = dict()

        # TODO - explain how you can pass a dict of darkness values, not just a float (and use case for this - black planets)
        if isinstance(self.args.darkness, float):
            darkness_dict = {color_name: self.args.darkness for color_name in self.palette.keys()}
        elif isinstance(self.args.darkness, dict):
            darkness_dict = self.args.darkness
            if "*" in darkness_dict:
                star_value = darkness_dict.pop("*")
                for color_name in self.palette.keys():
                    darkness_dict[color_name] = darkness_dict.get(color_name, star_value)

        mono_image_dict = {
            color: blur_image(mono_image, self.args.blur_rad) for color, mono_image in self.mono_images_dict.items()
        }

        # Setting a random seed at the start of this function ensures the lines will be the same (unless params change)
        global_random_seed(self.args.seed)

        # Create rich progress bars for each color
        progress = myprogress(verbose=verbose)
        with progress:
            for color_idx, color_name in enumerate(self.palette.keys()):
                # Setup variables (including the place we'll start)
                n_lines = self.args.n_lines_per_color[color_idx]
                m_image = mono_image_dict[color_name]
                # > line_dict[color_name] = []
                i = np.random.choice(list(self.args.d_joined.keys())).item()  # First node

                # Add task for this color
                task = progress.add_task(f"{color_name}", total=n_lines)

                for n in range(n_lines):  # range(n_lines): #, leave=False):
                    # Choose and add line
                    j = choose_and_subtract_best_line(
                        m_image=m_image,
                        i=i,
                        w=self.w,
                        n_random_lines=self.args.n_random_lines,
                        darkness=darkness_dict[color_name],
                        d_joined=self.args.d_joined,
                        t_pixels=self.args.t_pixels,
                    )
                    # > line_dict[color_name].append((i, j))
                    yield color_name, i, j

                    # Get the outgoing node (optionally jumping to a random non-consecutive node, for svg security)
                    i = j + 1 if (j % 2 == 0) else j - 1
                    if self.args.n_consecutive != 0 and ((n + 1) % self.args.n_consecutive) == 0:
                        i = list(self.args.d_joined.keys())[t.randint(0, len(self.args.d_joined), (1,))]

                    # Update progress bar
                    progress.update(task, advance=1)

                # Make sure progress bar has completed
                progress.update(task, completed=n_lines)

        # If not verbose then we don't have a progress bar, just a single printout at the end
        if not verbose:
            print(f"Created canvas in {time.time() - t0:.2f} seconds")

        # Return line dict
        return line_dict

    # Creates images / animations from the art
    def paint_canvas(
        self,
        line_dict: list[tuple[int, int]],
        x_output: int | None = None,
        rand_perm: float = 0.0025,
        fraction: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
        background_color: Tuple[int, int, int] | None = (0, 0, 0),
        show_individual_colors: bool = False,
        color_substitution: dict[str, tuple[int, int, int]] = {},
        animation: dict[str, int] | None = None,  # lines_per_frame, frame_duration
        line_width_multiplier: float = 1.0,
        png: bool = True,
        verbose: bool = False,
        make_html: bool = True,
        html_line_width: float = 0.12,
        html_x: int = 800,
        html_x_small: int | None = None,
        html_total_slider_steps: int = 150,
        html_rand_perm: float = 0.0025,
        html_bg_color: tuple[int, int, int] = (0, 0, 0),
    ):
        """
        Takes the line_dict, and uses it to create an svg of the output, then saves it
        """
        t0 = time.time()

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
                line_width=html_line_width,
                steps_per_slider=html_total_slider_steps,
                rand_perm=html_rand_perm,
                bg_color=html_bg_color,
            )
            with open(f"outputs/{self.args.name}/{self.args.name}.html", "w") as f:
                f.write(html)

        # Possibly sub a color for a different one
        color_dict = {k: v for k, v in self.palette.items()}
        for k, v in color_substitution.items():
            assert k in color_dict, f"Color {k} not in palette."
            color_dict[k] = v

        # If group orders are integers, they should be turned into strings
        if isinstance(self.args.group_orders, str):
            group_orders_str = self.args.group_orders
        else:
            single_color_batch = "".join([color_name[0] for color_name in color_dict])
            group_orders_str = single_color_batch * self.args.group_orders

        # Now that we have group orders as a string, turn it into a list of color indices
        d = {color_name[0]: idx for idx, color_name in enumerate(color_dict.keys())}
        group_orders_list = [d[char] for char in group_orders_str]

        # Get progress bar stuff
        progress_bar = tqdm_notebook(
            total=sum([len(lines) for lines in line_dict.values()]),
            desc="Painting canvas",
            disable=not verbose,
        )
        line_counter = 0
        frame_counter = 0

        # Get x and y values, and also the coords dict
        if x_output is None:
            x_output = self.x
        y_output = int(self.y * x_output / self.x)
        d_coords = build_through_pixels_dict(
            x_output,
            y_output,
            self.args.n_nodes,
            shape=self.args.shape,
            critical_distance=14,
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

            bg_color = [0.0, 0.0, 0.0, 0.0] if background_color is None else [c / 255 for c in background_color]
            context.set_source_rgba(*bg_color)
            context.paint()

            for i_idx, i in enumerate(group_orders_list):
                color_name = list(color_dict.keys())[i]
                color_value = color_dict[color_name]
                lines = line_dict[color_name]
                context.set_source_rgb(*[c / 255 for c in color_value])

                n_groups = len([j for j in group_orders_list if j == i])
                group_order = len([j for j in group_orders_list[:i_idx] if j == i])

                n = int(len(lines) / n_groups)
                lines_to_draw = lines[::-1][n * group_order : n * (group_order + 1)]
                # print(f"Setting color to {[c/255 for c in color_value]} for {len(lines_to_draw)} lines.")
                # print(f"{i_idx+1:2}/{len(group_orders_list)}: {len(lines_to_draw):4} {color_name}")

                # if verbose:
                #     print(f"{i_idx + 1:2}/{len(group_orders_list)}: {len(lines_to_draw):4} {color_name}")

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

                    if (animation is not None) and (line_counter % animation["lines_per_frame"] == 0):
                        context.stroke()
                        gif_png_filename = str(self.save_dir / f"animation_{frame_counter:03}.png")
                        surface.write_to_png(gif_png_filename)
                        frame_counter += 1
                        progress_bar.update(line_counter - progress_bar.n)

                    line_counter += 1
                progress_bar.update(line_counter - progress_bar.n)

                context.stroke()

            # Loop through all the files used for animations, and save them
            if animation is not None:
                png_filenames = sorted(self.save_dir.glob("animation_*.png"))

                # Create gif & mp4
                progress_bar_2 = tqdm_notebook(total=len(png_filenames), desc="Creating gif")
                frames_per_second = 1000 / animation["frame_duration"]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
                video_output = cv2.VideoWriter(
                    str(self.save_dir / f"{img_name}.mp4"),
                    fourcc,
                    frames_per_second,
                    (x_output, y_output),
                )

                # with imageio.get_writer(str(self.save_dir / f"{img_name}.gif"), mode='I', duration=animation["frame_duration"], loop=0) as writer:
                for png_file in png_filenames:
                    progress_bar_2.update(1)
                    # # Gif stuff
                    # image = imageio.imread(str(png_file))
                    # writer.append_data(image) # type: ignore
                    # Video stuff
                    frame = cv2.imread(str(png_file))
                    video_output.write(frame)
                    # Remove file
                    png_file.unlink()

                video_output.release()
                cv2.destroyAllWindows()

            # Get a final PNG, if necessary
            if png:
                surface.write_to_png(str(self.save_dir / f"{img_name}.png"))

        if show_individual_colors:
            for color_name, color_value in color_dict.items():
                lines = line_dict[color_name]
                with cairo.SVGSurface(
                    str(self.save_dir / f"{img_name}_{color_name}.svg"),
                    x_output,
                    y_output,
                ) as surface:
                    context = cairo.Context(surface)
                    context.scale(x_output, y_output)
                    context.set_line_width(0.0002 * line_width_multiplier)
                    # Set background color either black or white, whichever is more appropriate
                    use_black_background = (sum(color_value) >= 255 * 2) or (color_name == "white")
                    bg_color = [0.0, 0.0, 0.0] if use_black_background else [1.0, 1.0, 1.0]
                    # print(color_name, sum(color_value), bg_color)  # TODO - remove
                    context.set_source_rgb(*bg_color)
                    context.paint()
                    context.set_source_rgb(*[c / 255 for c in color_value])

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

    def generate_thread_art_html(
        self,
        line_dict: dict[tuple[int, int], list[tuple[int, int]]],
        x: int = 800,
        x_small: int | None = None,
        line_width: float = 0.12,
        steps_per_slider: int = 150,
        rand_perm: float = 0.0025,
        bg_color: tuple[int, int, int] = (0, 0, 0),
    ) -> str:
        d = {color_name[0]: idx for idx, color_name in enumerate(self.args.palette.keys())}
        group_orders_list = [d[char] for char in self.args.group_orders]
        group_orders_total = {i: group_orders_list.count(i) for i in set(group_orders_list)}
        group_orders_count = {i: 0 for i in set(group_orders_list)}

        # Calculate dimensions
        full_width = x
        full_height = int(full_width * self.args.y / self.args.x)

        # Individual color images will be half the scale
        x_small = x_small or int(x / 3)
        small_width = int(x_small)
        small_height = int(small_width * self.args.y / self.args.x)

        data = {
            "line_dict": {k: v[::-1] for k, v in line_dict.items()},
            "d_coords": {int(k): [round(c, 2) for c in v.tolist()] for k, v in self.args.d_coords.items()},
            "palette": self.args.palette,
            "group_orders_list": group_orders_list,
            "group_orders_total": group_orders_total,
            "group_orders_count": group_orders_count,
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

const data = {json.dumps(data)};
console.log(data);

{load_template("init.js")}
</script>
"""


def load_template(filename: str) -> str:
    path = Path(__file__).parent / "templates" / filename
    assert path.exists()
    return path.read_text()


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
    if mode == "linear":
        return linear_blur_image(img, rad, **kwargs)

    elif mode == "gaussian":
        # We need to go through torch -> numpy -> image (filter) -> numpy -> torch
        return t.from_numpy(np.asarray(Image.fromarray(img.numpy()).filter(ImageFilter.GaussianBlur(radius=rad)))).to(
            t.float
        )

    else:
        raise ValueError("Mode must be either 'linear' or 'gaussian'.")


# Generates a bunch of random lines and chooses the best one
def choose_and_subtract_best_line(
    m_image: Tensor,
    i: int,
    w: Optional[Tensor],
    n_random_lines: int | Literal["all"],
    darkness: float,
    d_joined: dict[int, list[int]],
    t_pixels: Tensor,
) -> int:
    """
    Generates a bunch of random lines (choosing them from `d_joined` which is a dictionary mapping node ints to all the
    nodes they're connected to), picks the best line, subtracts its darkness from the image, and returns that line.
    """
    # Choose `j` random lines
    if isinstance(n_random_lines, int):
        if n_random_lines < len(d_joined[i]):
            j_choices = t.from_numpy(
                np.random.choice(d_joined[i], min(len(d_joined[i]), n_random_lines), replace=False)
            ).long()
        else:
            j_choices = t.tensor(d_joined[i]).long()
    elif n_random_lines == "all":
        j_choices = t.tensor(d_joined[i]).long()
    else:
        raise ValueError(f"Unexpected value {n_random_lines=}")
    n_lines = j_choices.size(0)

    # Get the pixels in the line, and rearrange it
    coords_yx = t_pixels[i, j_choices.tolist()].long()  # [n_lines 2 pixels]
    is_zero = coords_yx.sum(dim=1) == 0  # [n_lines pixels]
    coords_yx = einops.rearrange(coords_yx, "j yx pixels -> yx (j pixels)")  # [2 n_lines*pixels]

    # Get the pixels in the line, and reshape it back to [n_lines, pixels]
    # TODO - I should be able to index without flattening it first, to make this code much shorter
    pixel_values = m_image[coords_yx[0], coords_yx[1]]  # [n_lines*pixels]
    pixel_values = einops.rearrange(pixel_values, "(j pixels) -> j pixels", j=n_lines).masked_fill(is_zero, 0)

    # Optionally index & rearrange the weighting, in the same way as the pixels
    if isinstance(w, Tensor):
        w_pixel_values = w[coords_yx[0], coords_yx[1]]
        w_pixel_values = einops.rearrange(w_pixel_values, "(j pixels) -> j pixels", j=n_lines).masked_fill(is_zero, 0)

        w_sum = w_pixel_values.sum(dim=-1)  # [n_lines]
        scores = (pixel_values * w_pixel_values).sum(-1) / w_sum  # [n_lines]

    else:
        lengths = (~is_zero).sum(-1)  # [n_lines]
        scores = pixel_values.sum(-1) / lengths  # [n_lines]

    best_j = j_choices[scores.argmax()].item()
    assert isinstance(best_j, int)

    coords_yx = t_pixels[i, best_j].long()  # [yx=2 pixels]
    is_zero = coords_yx.sum(0) == 0  # [pixels]
    coords_yx = coords_yx[:, ~is_zero]
    m_image[coords_yx[0], coords_yx[1]] -= darkness

    return best_j


# Permutes coordinates, to stop weird-looking line pattern effects (used by `paint_canvas` function)
def hacky_permutation(y, x, r):
    R = r * (2 * np.random.random() - 1)

    if (x < 0.01) or (x > 0.99):
        return y + R, x
    else:
        return y, x + R
