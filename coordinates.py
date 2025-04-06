"""
Includes all funcs related to coordinate placement
"""

import gc
import random

import numpy as np
import torch as t
from IPython.display import clear_output
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from misc import get_size_mb

t.classes.__path__ = []

# ================================================================


# Returns distance between two points
def get_distance(p0: np.ndarray, p1: np.ndarray):
    """
    Given two coordinates, returns distance between them
    """

    δ = np.subtract(p1, p0)
    distance = (δ**2).sum() ** 0.5

    return distance
    # return (distance, δ / distance) if return_δ else distance


# Truncates a single pair of coordinates, to make sure that none are outside the bounds of the image (used in `build_through_pixels_dict` function)
def truncate_coords(coords, limits):
    """
    Truncates coordinates at some limit
    """

    for i in range(2):
        coords[i] = max(0, min(coords[i], limits[i]))

    return coords


# Truncates an array of pixel coordinates, to make sure that none are outside the bounds of the image (used in `build_through_pixels_dict` function)
def truncate_pixels(pixels, limits):
    """
    Truncates a pixels array (i.e. that was generated from the through_pixels function)

    This is different from truncate_coords because we can actually make this thing shorter.
    """

    # for i in range(2):
    #     pixels[i][pixels[i] < 0] = 0
    #     pixels[i][pixels[i] > limits[i]] = limits[i]

    mask = (pixels[0] >= 0) & (pixels[0] <= limits[0]) & (pixels[1] >= 0) & (pixels[1] <= limits[1])
    pixels = pixels[:, mask]

    return pixels


# Gets array of pixels going through any two points (used in lots of other functions)
def through_pixels(p0: Float[Tensor, "2"], p1: Float[Tensor, "2"], step_size: float = 1.0) -> Float[Tensor, "2 len"]:
    """
    Given two PyTorch tensors p0 and p1, returns the pixels that the line connecting p0 & p1 passes through.

    Returns it as lower-resolution floats for further processing.
    """

    assert isinstance(p0, t.Tensor) and isinstance(p1, t.Tensor), "Inputs must be PyTorch tensors."
    assert p0.shape == (2,) and p1.shape == (2,), "Inputs must be 1D tensors of shape (2,)."

    δ = p1 - p0
    distance = t.sqrt((δ**2).sum())

    assert distance > 0, f"Error: {p0} and {p1} have distance zero."

    num_steps = int(distance / step_size) + 1
    pixels_in_line = p0 + t.outer(t.linspace(0, 1, num_steps, dtype=t.float32, device=p0.device), δ)

    return pixels_in_line.T


def get_thick_line(p0, p1, all_coords, thickness=1):
    p0y, p0x = p0
    p1y, p1x = p1

    if p0x == p1x:
        a = 0
        b = 1
    else:
        a = (p1y - p0y) / (p1x - p0x)
        ab_norm = np.sqrt(1 + a**2)
        a /= ab_norm
        b = -1 / ab_norm

    c_p = a * p0x + b * p0y
    c_q = a * all_coords[1] + b * all_coords[0]

    return all_coords[:, np.abs(c_p - c_q) < thickness]


def pair_to_index(
    i: int,
    j: int | Int[Tensor, "batch"],
    n: int,
) -> int | Tensor:
    """
    Maps a tuple (i, j) where 0 <= i < j < n to a unique index
    in the range [0, 0.5*n*(n-1) - 1].

    This function creates a bijective mapping from all possible (i, j) pairs
    to a continuous range of indices.

    We can also pass `i` and/or `j` as tensors, in which case the function will return a tensor of indices.

    Example:
        For n = 4, the pairs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        will be mapped to indices 0, 1, 2, 3, 4, 5 respectively.
    """
    # Convert scalar tensors / arrays to plain integers
    if hasattr(j, "item"):
        j = j.item() if j.size == 1 else j

    # Simple case: just return the index as an integer, if both are ints
    if isinstance(j, int):
        i, j = min(i, j), max(i, j)

    # Messier case where j is an array so we need to do some casting and tensor sorting
    else:
        if not isinstance(j, t.Tensor):
            j = t.tensor(j, dtype=t.int16)
        i = t.tensor(i, dtype=t.int16).repeat(len(j))
        i, j = t.min(i, j), t.max(i, j)

    return (n - 1) * i - i * (i + 1) // 2 + j - 1


# pairs = [(0, 1), (0, 2), (1, 2)]
# n = 3

# for index, (i, j) in enumerate(pairs):
#     assert pair_to_index(i, j, n) == index


# ================================================================


def build_through_pixels_dict(
    x,
    y,
    n_nodes,
    shape: str,
    critical_distance: int = 14,
    only_return_d_coords: bool = False,
    width_to_gap_ratio: float = 1.0,
    step_size: float = 1.0,
    n_lines_per_memory_clear: int | None = None,  # set to about 5k?
    debug: bool = False,
):
    if shape == "Rectangle" and isinstance(n_nodes, int):
        assert (n_nodes % 4) == 0, f"n_nodes = {n_nodes} needs to be divisible by 4, or else there will be an error"

    d_coords: dict[int, Float[Tensor, "2"]] = {}  # maps i -> coordinates of node i on perimeter
    d_joined: dict[int, list[int]] = {}  # maps node index i -> list of nodes connected to that one
    d_sides: dict[int, list[int]] = {}  # maps side index j -> list of nodes on that side

    d_archetypes = {}  # see later in code

    # Note - we used to have `d_pixels` which mapped tuples of f(i, j) -> Int[Tensor, "2 len"], but now we just use a
    # tensor `t_pixels` of size (0.5 * n_nodes * (n_nodes + 1), 2, max_pixels), where the first dimension is the one
    # we use to index some pair f(i, j) into the tensor. This is faster in the actual generation phase cause we get to
    # just index once and concatenate, and do everything as a single tensor. Note, the use of `max_pixels` means there's
    # some buffer room in the tensor (which we pad with zeros), but this small memory inefficiency is basically fine.
    # Note, f(i, j) is the function pair_to_index above.
    max_pixels_guess = int((x**2 + y**2) ** 0.5 / step_size) + 2
    n_lines_total = int(0.5 * n_nodes * (n_nodes - 1))
    t_pixels = t.zeros((n_lines_total, 2, max_pixels_guess), dtype=t.int16)

    if shape == "Rectangle":
        # we either read the number of nodes and divide them proportionally between sides, or the number of nodes is
        # externally specified
        if type(n_nodes) in [int, np.int64]:
            nx = 2 * int(n_nodes * 0.25 * x / (x + y))
            ny = 2 * int(n_nodes * 0.25 * y / (x + y))

            while 2 * (nx + ny) < n_nodes:
                if ny >= nx:
                    ny += 2
                else:
                    nx += 2
            while 2 * (nx + ny) > n_nodes:
                if ny >= nx:
                    ny -= 2
                else:
                    nx -= 2
        elif isinstance(n_nodes, tuple):
            nx, ny = n_nodes
            n_nodes = 2 * (nx + ny)
        else:
            raise TypeError(f"n_nodes = {n_nodes} is of type {type(n_nodes)}, which is not supported")

        nodes_per_side_list = [ny, nx, ny, nx]

        starting_idx_list = np.cumsum([0] + nodes_per_side_list).tolist()

        x -= 1
        y -= 1

        xd = x / nx
        yd = y / ny
        X0_list = t.tensor([(y, x), (0, x), (0, 0), (y, 0)])

        Xd_list = t.tensor([(-yd, 0), (0, -xd), (yd, 0), (0, xd)])
        n0, n1, n2, n3, n4 = starting_idx_list

        # =============== get all the coordinates of the nodes, and their sides ===============

        for side, starting_idx, X0, Xd in zip(range(4), starting_idx_list, X0_list, Xd_list):
            nodes_per_side = nodes_per_side_list[side]

            δ = (width_to_gap_ratio - 1) / (2 * (width_to_gap_ratio + 1))

            for i in range(nodes_per_side):
                idx = starting_idx + i
                coords_raw = X0 + (i + 0.5) * Xd
                if (i % 2) == 0:
                    coords_raw -= δ * Xd
                else:
                    coords_raw += δ * Xd
                d_coords[idx] = truncate_coords(coords_raw, limits=[y, x])
                d_sides[idx] = side

        if only_return_d_coords:
            return d_coords

        # =============== get the joined pixels (i.e. the ones not on the same side) ===============

        for i in d_sides:
            d_joined[i] = [j for j in range(n4) if d_sides[i] != d_sides[j]]

        # =============== compute archetypal pixels and fill the pixel tensor ===============

        progress_bar = tqdm(
            desc="Building pixels dict",
            # total=sum([len(d_joined[i]) for i in d_joined]) // 2,
            total=len(d_joined),
        )

        # Build archetypes only when needed and store them in a dictionary
        def get_archetype(key_type, a, b=None):
            key = f"{key_type}_{a}" if b is None else f"{key_type}_{a}_{b}"
            if key not in d_archetypes:
                if key_type == "vertical":
                    i, j = n2, (n3 + a) % n4
                elif key_type == "horizontal":
                    i, j = n1, n2 + a
                elif key_type == "diagonal":
                    if nx >= ny:
                        i, j = n2 + a, n2 - b
                    else:
                        i, j = n2 - a, n2 + b
                d_archetypes[key] = through_pixels(d_coords[i], d_coords[j], step_size=step_size)
            return d_archetypes[key].clone()

        for idx, i in enumerate(d_joined):
            for j in d_joined[i]:
                if i >= j:
                    continue

                # === first, check if they're opposite vertical, if so then populate using the archetypes ===
                if (d_sides[i], d_sides[j]) == (1, 3):
                    # this makes sure the node with vertex 0 is seen as being on side 3, not side 0
                    if i == 0:
                        i_, j_ = j, n4
                    else:
                        i_, j_ = i, j
                    δ = (i_ + j_) - (2 * n2 + ny)
                    pixels = get_archetype("vertical", abs(δ))
                    if δ < 0:
                        pixels[1] = -pixels[1]
                    pixels[1] += (n2 - i_) * xd
                    pixels_truncated = truncate_pixels(pixels.to(t.int16), [y, x])
                    t_pixels[pair_to_index(i, j, n_nodes), :, : pixels_truncated.size(1)] = pixels_truncated

                # === then, check if they're opposite horizontal, if so then populate using the archetypes ===
                elif (d_sides[i], d_sides[j]) == (0, 2):
                    δ = (i + j) - (2 * n1 + nx)
                    pixels = get_archetype("horizontal", abs(δ))
                    if δ < 0:
                        pixels[0] = -pixels[0]
                    pixels[0] += (n1 - i) * yd
                    pixels_truncated = truncate_pixels(pixels.to(t.int16), [y, x])
                    t_pixels[pair_to_index(i, j, n_nodes), :, : pixels_truncated.size(1)] = pixels_truncated

                # === finally, the diagonal case ===
                else:
                    i_side = d_sides[i]
                    j_side = d_sides[j]

                    x_side = i_side if (i_side % 2 == 1) else j_side
                    y_side = i_side if (i_side % 2 == 0) else j_side

                    if i_side == 0 and j_side == 3:
                        i_, j_ = j, i
                        i_side, j_side = 3, 0
                    else:
                        i_, j_ = i, j

                    i_len = starting_idx_list[i_side + 1] - i_
                    j_len = j_ - starting_idx_list[j_side]

                    x_len = i_len if (i_side % 2 == 1) else j_len
                    y_len = i_len if (i_side % 2 == 0) else j_len

                    adj = min(i_len, j_len)
                    opp = max(i_len, j_len)

                    pixels = get_archetype("diagonal", adj, opp)

                    # flip in x = y
                    if ((x_len > y_len) != (x > y)) and (x_len != y_len):
                        pixels = pixels.flip(0)
                    # flip in x
                    if x_side == 3:
                        pixels[0] = y - pixels[0]
                    # flip in y
                    if y_side == 0:
                        pixels[1] = x - pixels[1]

                    pixels_truncated = truncate_pixels(pixels.to(t.int16), [y, x])
                    t_pixels[pair_to_index(i, j, n_nodes), :, : pixels_truncated.size(1)] = pixels_truncated

                # progress_bar.update(1)
                if (n_lines_per_memory_clear is not None) and (random.random() < 1 / n_lines_per_memory_clear):
                    gc.collect()
            progress_bar.update(1)

        # progress_bar.n = sum([len(d_joined[i]) for i in d_joined]) // 2
        progress_bar.n = len(d_joined)

    elif shape == "Ellipse":
        assert x % 2 == 0, "x must be even to take advantage of symmetry"

        angles = np.linspace(0, 2 * np.pi, n_nodes + 1)[:-1]

        x_coords = 1 + ((0.5 * x) - 2) * (1 + np.cos(angles))
        y_coords = 1 + ((0.5 * y) - 2) * (1 - np.sin(angles))

        coords = t.stack([t.from_numpy(y_coords), t.from_numpy(x_coords)]).T

        d_sides = None
        d_joined = {n: [] for n in range(n_nodes)}
        for i, coord in enumerate(coords):
            d_coords[i] = coord
            # the line below is an efficient way of saying "d_joined[i] = all nodes at least `critical_distance` from i"
            d_joined[i] = sorted(
                np.mod(
                    range(i + critical_distance, i + (n_nodes + 1 - critical_distance)),
                    n_nodes,
                )
            )

        # The second half are added via symmetry
        # total = sum([len(d_joined[i]) for i in d_joined]) // 4
        total = len(d_joined)
        progress_bar = tqdm(desc="Building pixels dict", total=total)

        for i1 in d_joined:
            p1 = d_coords[i1]
            for i0 in d_joined[i1]:
                # Avoid double counting: only consider (i0, i1) for i0 < i1
                if i0 > i1:
                    break

                # Check if the reflection of this line is already in the dict
                idx = pair_to_index(i0, i1, n_nodes)
                reflection_idx = pair_to_index(n_nodes + 1 - i1, n_nodes + 1 - i0, n_nodes)
                if t_pixels[reflection_idx].max() > 0:
                    y_reflected, x_reflected = t_pixels[reflection_idx]
                    pixels = t.stack([(y - y_reflected).flip(0), x_reflected.flip(0)])

                # If reflection isn't in the dict, we need to create it
                else:
                    p0 = d_coords[i0]
                    pixels = through_pixels(p0, p1, step_size=step_size)

                pixels_truncated = truncate_pixels(pixels.to(t.int16), [y - 1, x - 1])
                t_pixels[idx, :, : pixels_truncated.size(1)] = pixels_truncated

                # progress_bar.update(1)
                if (n_lines_per_memory_clear is not None) and (random.random() < 1 / n_lines_per_memory_clear):
                    gc.collect()
            progress_bar.update(1)

        if only_return_d_coords:
            return d_coords

    # Print the estimated size in MB of each dictionary
    sizes = {
        "d_coords": get_size_mb(d_coords),
        "d_joined": get_size_mb(d_joined),
        "d_sides": get_size_mb(d_sides),
        "d_archetypes": get_size_mb(d_archetypes),
        "t_pixels": get_size_mb(t_pixels),
    }
    print("\nObject sizes in MB:")
    print("-" * 30)
    for obj_name, size in sizes.items():
        print(f"{obj_name}: {size:.4f} MB")
    print(f"Total: {sum(sizes.values()):.4f} MB")

    # =============== populate the tensor ===============

    # We overestimated to get the size of t_pixels, so we need to truncate it
    t_pixels_sum = t_pixels.sum(dim=(0, 1))
    max_pixels = t_pixels_sum.nonzero()[-1].item()
    print(f"Cropping {t_pixels.shape[-1] - max_pixels} pixels")
    t_pixels = t_pixels[:, :, :max_pixels]

    if not debug:
        clear_output()

    return d_coords, d_joined, d_sides, t_pixels
