"""
Includes all funcs related to coordinate placement
"""

import gc
import random

import matplotlib.pyplot as plt
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


'''def build_through_pixels_dict(
    x,
    y,
    n_nodes,
    shape: str,
    critical_fracs: tuple[float, float | None] = (0.02, None),
    only_return_d_coords: bool = False,
    width_to_gap_ratio: float = 1.0,
    step_size: float = 1.0,
    make_symmetric: bool = True,
    debug: bool = False,
) -> dict[int, Tensor] | tuple[dict[int, Tensor], dict[int, list[int]], dict[int, list[int]], Tensor]:
    """
    Args:
        x: width of the image
        y: height of the image
        n_nodes: number of nodes in the image
        shape: either "Rectangle" or "Ellipse", for rectangle / circular images
        critical_fracs: minimum distance between nodes that are considered "connected". Any nodes closer than this
            aren't connected. This is useful for the gantry threading, because it finds these kinds of lines hard. Note
            this is a tuple, referring to the strict and lenient fractions respectively (the former is the strict one
            because it refers to the kind of lines where the string crosses over itself; the latter is merely a sharp
            angle).
        only_return_d_coords: if True, only returns the d_coords dictionary, not the pixel tensor. This is used when
            we're painting the canvas (the tensor is a more useful form when generating the image).
        width_to_gap_ratio: ratio of the width of the gap to the width of the line. This makes sure the image looks
            accurate to physical representation.
        step_size: size of the step between pixels. Making this larger than 1 results in a quicker algorithm, but can
            lose accuracy past a certain point.
        make_symmetric: Experimental, makes connecting lines symmetric (I think the problem happens when we can draw
            a line but then not go back the same way with a similar line).
        debug: if True, we don't clear the output at the end of the function. This is useful for debugging.

    """
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
    max_distance = (x**2 + y**2) ** 0.5 if shape == "Rectangle" else max(x, y)
    max_pixels_guess = int(max_distance / step_size) + 2
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

        for i, i_side in d_sides.items():
            d_joined[i] = [j for j in range(n4) if d_sides[i] != d_sides[j]]
            if critical_fracs is not None:
                # If this side is clockwise, we can't sharply move anticlockwise (or else it'll cross over itself). For
                # example, use coord system (x, y) with (0, 0) at top-left: if we're on the clockwise side of a hook on
                # the top side with coords (10, 0) then moving to something like (0, 2) would be really bad because the
                # reverse of that move is tight & crosses over itself. We also shouldn't sharply move clockwise, but the
                # restriction here is a bit more relaxed.
                n_nodes_on_adj_side = nodes_per_side_list[(i_side + 1) % 4]
                ac_corner = starting_idx_list[(i_side + 1) % 4]  # node at the corner anticlockwise to `i`
                c_corner = starting_idx_list[i_side]  # node at the corner clockwise to `i`
                assert c_corner <= i < (ac_corner or n4), f"Error: {c_corner=}, {i=}, {ac_corner=}"

                banned_anticlockwise = range(
                    ac_corner, ac_corner + min(n_nodes_on_adj_side, int((ac_corner - i) * critical_fracs[i % 2]))
                )
                banned_clockwise = range(
                    c_corner - min(n_nodes_on_adj_side, int((i - c_corner) * critical_fracs[1 - i % 2])), c_corner
                )
                d_joined[i] = sorted(set(d_joined[i]) - set(banned_anticlockwise) - set(banned_clockwise))

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

            progress_bar.update(1)

        # progress_bar.n = sum([len(d_joined[i]) for i in d_joined]) // 2
        progress_bar.n = len(d_joined)

    elif shape == "Ellipse":
        assert x % 2 == 0, "x must be even to take advantage of symmetry"

        angles = np.linspace(0, 2 * np.pi, n_nodes + 1)[:-1]

        # Offset the angles by the width to gap ratio
        δ = (width_to_gap_ratio - 1) / (2 * (width_to_gap_ratio + 1))
        angle_diff = angles[1] - angles[0]
        angles[::2] += angle_diff * δ
        angles[1::2] -= angle_diff * δ
        angles = np.mod(angles, 2 * np.pi)

        x_coords = 1 + ((0.5 * x) - 2) * (1 + np.cos(angles))
        y_coords = 1 + ((0.5 * y) - 2) * (1 - np.sin(angles))

        coords = t.stack([t.from_numpy(y_coords), t.from_numpy(x_coords)]).T

        # Critical fraction gets converted to a number of nodes, i.e. all lines should be >= this distance (we
        # can do this because we assume symmetry in the ellipse)
        critical_n_nodes = [int(c * n_nodes) for c in critical_fracs]

        d_sides = None
        d_joined = {n: [] for n in range(n_nodes)}
        for i, coord in enumerate(coords):
            d_coords[i] = coord

            # Iterate around the nodes anticlockwise, with restrictions:
            #   - if abs(angle) < critical_frac, then we skip it
            #   - if abs(angle) < critical_frac_for_hook_sides, then we only do one side of it
            # So we have 2 possible reasons to add a node to the list:
            #   - abs(angle) > critical_frac_for_hook_sides
            #   - abs(angle) is in [critical_frac, critical_frac_for_hook_sides] range, and it's the correct side

            critical_n_nodes_start, critical_n_nodes_end = critical_n_nodes[:: 1 if i % 2 == 0 else -1]
            d_joined[i] = sorted(
                np.mod(range(i + critical_n_nodes_start + 1, i + (n_nodes - critical_n_nodes_end)), n_nodes)
            )

        # The second half are added via symmetry
        # total = sum([len(d_joined[i]) for i in d_joined]) // 4
        total = len(d_joined)
        progress_bar = tqdm(desc="Building pixels dict", total=total)

        for i1 in d_joined:
            p1 = d_coords[i1]
            for i0 in d_joined[i1]:
                # # Avoid double counting: only consider (i0, i1) for i0 < i1
                # if i0 > i1:
                #     break

                # # Check if the reflection of this line is already in the dict
                idx = pair_to_index(i0, i1, n_nodes)
                reflection_idx = pair_to_index(n_nodes - i1, n_nodes - i0, n_nodes)
                if t_pixels[reflection_idx].max() > 0:
                    y_reflected, x_reflected = t_pixels[reflection_idx]
                    pixels = t.stack([(y - y_reflected).flip(0), x_reflected.flip(0)])

                # If reflection isn't in the dict, we need to create it
                else:
                    p0 = d_coords[i0]
                    pixels = through_pixels(p0, p1, step_size=step_size)

                pixels_truncated = truncate_pixels(pixels.to(t.int16), [y - 1, x - 1])
                t_pixels[idx, :, : pixels_truncated.size(1)] = pixels_truncated

            progress_bar.update(1)

        if only_return_d_coords:
            return d_coords

    # We overestimated to get the size of t_pixels, so we need to truncate it
    t_pixels_sum = t_pixels.sum(dim=(0, 1))
    max_pixels = t_pixels_sum.nonzero()[-1].item()
    t_pixels_cropped = t_pixels[:, :, :max_pixels]

    # Turn d_joined into a symmetric dict
    if make_symmetric:
        for i, j_list in d_joined.items():
            d_joined[i] = [j for j in j_list if i in d_joined[j]]

    if debug:
        # > Print the estimated size in MB of each dictionary
        sizes = {
            "d_coords": get_size_mb(d_coords),
            "d_joined": get_size_mb(d_joined),
            "d_sides": get_size_mb(d_sides),
            "d_archetypes": get_size_mb(d_archetypes),
            "t_pixels": get_size_mb(t_pixels),
            "t_pixels_cropped": get_size_mb(t_pixels_cropped),
        }
        print("\nObject sizes in MB:")
        print("-" * 30)
        for obj_name, size in sizes.items():
            print(f"{obj_name}: {size:.4f} MB")
        print(f"Total: {sum(sizes.values()):.4f} MB")
        # > For 6 randomly chosen points, plot the lines that they connect to
        # Choose 6 random nodes
        selected_nodes = random.sample(list(d_coords.keys()), 6)
        fig, axes = plt.subplots(3, 2, figsize=(15, int(20 * y / x)))
        axes = axes.flatten()
        for i, node in enumerate(selected_nodes):
            ax = axes[i]
            # Plot the selected node (red)
            cy, cz = d_coords[node]
            ax.plot(cz, cy, "ro", markersize=8)
            # Plot its adjacent node (blue)
            adj_node = node + (1 if node % 2 == 0 else -1)
            cy_adj, cz_adj = d_coords[adj_node]
            ax.plot(cz_adj, cy_adj, "o", color="#00d7e1", markersize=6)
            # Plot connected nodes (black)
            if node in d_joined:
                for connected_node in d_joined[node]:
                    if connected_node in d_coords:
                        cy, cx = d_coords[connected_node]
                        ax.plot(cx, cy, "ko", markersize=4)
            # Remove all visual elements except dots
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
    else:
        clear_output()

    return d_coords, d_joined, d_sides, t_pixels_cropped '''
def build_through_pixels_dict(x, y, n_nodes, shape, critical_distance=14, only_return_d_coords=False, width_to_gap_ratio=1):
    """
    Clean version: each nail = 1 node (no parity).
    Returns: d_coords, d_pixels, d_joined, d_sides, t_pixels
    - d_coords[i] is a torch tensor [y, x] (int)
    - d_pixels[(i,j)] is a torch tensor shape (2, N) with (y,x) order
    - d_joined[i] is a sorted list of candidate j (excludes near neighbors within critical_distance along the ring)
    - d_sides[i] side index for rectangle (0..3) or 0 for ellipse
    - t_pixels is torch.zeros((n_nodes, n_nodes, 2, max_pixels), dtype=int)
    """
    import math
    from tqdm import tqdm
    import numpy as np
    import torch as t


    # sanity
    if shape == "Rectangle" and type(n_nodes) == int:
        assert (n_nodes % 4) == 0, f"n_nodes = {n_nodes} needs to be divisible by 4, or else there will be an error"


    # prepare containers
    d_coords = {}
    d_pixels = {}
    d_joined = {}
    d_sides = {}


    shape_l = shape.lower()


    # ---------- 1) build d_coords and d_sides ----------
    if shape_l in ("ellipse", "circle", "round"):
        # keep same convention as original: even x recommended
        assert x % 2 == 0, "x must be even to take advantage of symmetry"
        angles = - np.linspace(0, 2 * np.pi, n_nodes + 1)[:-1]
        x_coords = 1 + ((0.5 * x) - 2) * (1 + np.cos(angles))
        y_coords = 1 + ((0.5 * y) - 2) * (1 - np.sin(angles))
        coords = t.stack([t.from_numpy(y_coords).to(t.float32), t.from_numpy(x_coords).to(t.float32)]).T
        # store as int tensors (y,x)
        for i in range(n_nodes):
            arr = coords[i].round().to(t.int64)
            d_coords[i] = arr  # tensor([y, x])
            d_sides[i] = 0
    else:
        # Rectangle: distribute nodes along rectangle perimeter (clockwise)
        W = int(x)
        H = int(y)
        # lengths (approx) of edges (top,right,bottom,left)
        edge_lengths = [W, H, W, H]
        perim = sum(edge_lengths)
        for i in range(n_nodes):
            frac = i / n_nodes
            dist = frac * perim
            cur = 0.0
            placed = False
            for side_idx, L in enumerate(edge_lengths):
                if dist <= cur + L - 1e-9:
                    offset = dist - cur
                    if side_idx == 0:  # top
                        xx = int(round(offset))
                        yy = 0
                    elif side_idx == 1:  # right
                        xx = W - 1
                        yy = int(round(offset))
                    elif side_idx == 2:  # bottom
                        xx = int(round(W - 1 - offset))
                        yy = H - 1
                    else:  # left
                        xx = 0
                        yy = int(round(H - 1 - offset))
                    xx = max(0, min(W - 1, xx))
                    yy = max(0, min(H - 1, yy))
                    d_coords[i] = t.tensor([yy, xx], dtype=t.int64)
                    d_sides[i] = side_idx
                    placed = True
                    break
                cur += L
            if not placed:
                d_coords[i] = t.tensor([0, 0], dtype=t.int64)
                d_sides[i] = 0


    if only_return_d_coords:
        return d_coords


    # ---------- 2) build d_joined: allowed target nodes for each i ----------
    # rule: allow j if circular min distance >= critical_distance (avoid too-short neighbors)
    for i in range(n_nodes):
        allowed = []
        for j in range(n_nodes):
            if j == i:
                continue
            # circular distance along ring
            diff = abs(j - i)
            circ = min(diff, n_nodes - diff)
            if circ >= critical_distance:
                allowed.append(j)
        # sort by (circular) positive order starting after i to have deterministic ordering
        # we will sort by (j - i) % n_nodes to keep consistent traversal order
        allowed_sorted = sorted(allowed, key=lambda j: ((j - i) % n_nodes))
        d_joined[i] = allowed_sorted


    # ---------- 3) compute pixel lists for each valid pair (i,j) ----------
    # To save memory/time compute only for pairs present in d_joined.
    pairs = []
    for i in range(n_nodes):
        for j in d_joined[i]:
            # ensure we only compute once per unordered pair: compute when i < j
            if i < j:
                pairs.append((i, j))


    # progress bar optional
    progress_bar = tqdm(total=len(pairs), desc="Building pixels dict")


    # Helper: call existing through_pixels(p0, p1) which should return tensor (2,N) of (y,x)
    for (i0, i1) in pairs:
        p0 = d_coords[i0].to(t.int64)
        p1 = d_coords[i1].to(t.int64)
        # compute pixel line tensor (2, N)
        pixels = through_pixels(p0, p1).to(t.int64)  # reuse repo helper
        if pixels.numel() == 0:
            progress_bar.update(1)
            continue
        # store forward and reversed
        d_pixels[(i0, i1)] = pixels  # (2, N) where row0 = y, row1 = x
        d_pixels[(i1, i0)] = pixels.flip(-1)  # reverse order so (i1,i0) goes back
        progress_bar.update(1)


    progress_bar.close()


    # ---------- 4) Final cleanup: ensure d_joined only contains pairs we computed ----------
    for i in range(n_nodes):
        valid_js = [j for j in d_joined[i] if (i, j) in d_pixels]
        d_joined[i] = valid_js


    # ---------- 5) build t_pixels tensor (n_nodes, n_nodes, 2, max_pixels) ----------
    if len(d_pixels) == 0:
        # no pixels computed (unlikely) -> return empty tensor
        t_pixels = t.zeros((n_nodes, n_nodes, 2, 0), dtype=t.int64)
    else:
        max_pixels = max([p.size(1) for p in d_pixels.values()])
        t_pixels = t.zeros((n_nodes, n_nodes, 2, max_pixels), dtype=t.int64)
        for (i, j), pixels in d_pixels.items():
            L = pixels.size(1)
            t_pixels[i, j, :, :L] = pixels


    output = [d_coords, d_pixels, d_joined, d_sides, t_pixels]
    return output



# def node_distance(i: int, j: int, n_nodes: int, signed: bool = False) -> int | tuple[int, int]:
#     """Gets closest distance between nodes i and j, i.e. i + dist = j (for signed)."""

#     anticlockwise_dist = (j - i) % n_nodes
#     clockwise_dist = (i - j) % n_nodes

#     if signed:
#         return min(anticlockwise_dist, clockwise_dist)
#     else:
#         return anticlockwise_dist, clockwise_dist
