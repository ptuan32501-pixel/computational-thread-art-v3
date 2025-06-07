import itertools
import math
import os
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PyPDF2 import PdfMerger, PdfReader
from reportlab.lib.colors import black, blue, gray
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from coordinates import get_distance
from image_color import ROOT_PATH, Img
from misc import get_range_of_lines

ARCS = {"clockwise": "G2", "anticlockwise": "G3"}


def generate_thread_art_gcode(
    line_dict: dict[tuple[int, int, int], list[tuple[int, int]]],
    n_nodes: int,
    group_orders: str,
    starting_posn: tuple[float, float],
    debug_nodes: list[int] = [],
    feed_rate: int | None = None,
    arc_feed_rate: int | None = None,
    inner_radius: int = 10,
    outer_radius: int = 5,
    test_distance: int = 0,
) -> list[list[str]]:
    """
    Generates full thread art GCode, by breaking down each line dict color according to group orders.

    Args:
        line_dict: Dictionary with color names as keys and lists of lines as values
        n_nodes: Total number of nodes (hook sides) around the circular frame
        group_orders: String representing the order of groups for each color
        starting_posn: Starting position of the arm (x, y) in mm
        debug_nodes: List of noes we'll visit before starting, in order to debug
        feed_rate: Movement speed in mm/min
        inner_radius: How far inside the hook do we go before going outside and looping around it
        outer_radius: How far outside the hook do we go before looping around it (only applies if lines not arcs)
        test_distance: If supplied, we subtract this many mm from radius (so we can test going close to edge, not on it)
    """
    # # Gets the estimated center and radius of the circle, from the `coords` supplied
    # coords_normalized = {index / n_nodes: coord for index, coord in coords.items()}
    # radius, center, starting_angle = estimate_circle_from_coords(coords_normalized)
    # radius -= debug_radius  # Take off the debug radius, if any

    # Get the radius & starting angle (which is the angle of node 1.5, i.e. a gap between nodes)
    x, y = starting_posn
    starting_angle = math.atan2(y, x)
    radius = math.sqrt(x**2 + y**2) - test_distance

    colors = list(line_dict.keys())
    group_orders_list = [int(i) - 1 for i in group_orders.split(",")]

    gcode = []

    # Get the first lines, for debugging
    debug_lines = _generate_thread_art_gcode_single_line_group(
        line_list=[(i, j + (1 if j % 2 == 0 else -1)) for i, j in zip(debug_nodes, debug_nodes[1:])],
        n_nodes=n_nodes,
        radius=radius,
        starting_angle=starting_angle,
        feed_rate=feed_rate,
        arc_feed_rate=arc_feed_rate,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        use_origin=True,
    )
    gcode.append(debug_lines)

    for i_idx, i in enumerate(group_orders_list):
        lines = line_dict[colors[i]]
        lines = [line[::-1] for line in lines[::-1]]  # Reverse the order of the lines, as well as their directions

        # Count occurrences of this group in total and so far
        total_occurrences = group_orders_list.count(i)
        current_occurrence = group_orders_list[: i_idx + 1].count(i)

        # Calculate lines per occurrence, ensuring all lines are used
        base_lines_per_group = len(lines) // total_occurrences
        remainder = len(lines) % total_occurrences

        # Calculate start and end indices for this occurrence
        start_idx = (base_lines_per_group * (current_occurrence - 1)) + min(current_occurrence - 1, remainder)
        end_idx = (base_lines_per_group * current_occurrence) + min(current_occurrence, remainder)
        # print(colors[i], start_idx, end_idx)

        # Get the lines for this occurrence
        lines = _generate_thread_art_gcode_single_line_group(
            line_list=lines[start_idx:end_idx],
            n_nodes=n_nodes,
            radius=radius,
            starting_angle=starting_angle,
            feed_rate=feed_rate,
            arc_feed_rate=arc_feed_rate,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
        )
        gcode.append(lines)

    return gcode


def _generate_thread_art_gcode_single_line_group(
    line_list: list[tuple[int, int]],
    n_nodes: int,
    radius: float,
    starting_angle: float,
    feed_rate: int | None,
    arc_feed_rate: int | None,
    inner_radius: int = 10,
    outer_radius: int = 5,
    use_origin: bool = False,
) -> list[str]:
    """
    Generate G-code for thread art connecting points around a circular frame.

    It works by going to points exactly on the circumference of the circle with radius `radius` (these will be half way
    in between two hooks) and then loop in a small 180 degree arc around them. The line is designed to have no sharp
    turns, so the path between two hooks forms a smooth arc with the short path looping around the hooks.

    Args:
        line_list: List of tuples, each tuple (a, b) represents a thread line connecting hooks a and b
        n_nodes: Total number of nodes (hook sides) around the circular frame
        radius: Physical diameter of the circular frame in mm
        starting_angle: The angle at which we start, in radians (0 is at the right, 1.5 is at the top)
        feed_rate: Movement speed in mm/min, for the fast lines
        arc_feed_rate: Movement speed in mm/min, for the slower arcs around hooks
        inner_radius: How far inside the hook we move before going outside and looping around it
        outer_radius: How far outside the hook we go before looping around it (only applies if lines not arcs)
        use_origin: If True, we visit origin between each move (for debugging)
    """
    feed_rate_lines = f"F{feed_rate:.0f}" if feed_rate else ""
    feed_rate_arcs = f"F{arc_feed_rate:.0f}" if arc_feed_rate else feed_rate_lines

    # Check if we're flipping parity of the hooks
    if len(line_list) >= 2:
        line_list_diff_0 = line_list[0][1] - line_list[1][0]
        assert abs(line_list_diff_0) <= 1
        flip_hook_parity = line_list_diff_0 == 1
    else:
        flip_hook_parity = True

    # Get unit radius positions of all nodes (i.e. the edges of hooks)
    hook_angles = (np.arange(0, n_nodes) - 1.5) * 2 * math.pi / n_nodes + starting_angle

    # Initialize G-code with setup commands
    gcode = []
    gcode.append("; Thread Art G-code")
    gcode.append("; Generated for {} nodes on a {:.2f}mm radius circle".format(n_nodes, radius))
    gcode.append("; Copy to https://ncviewer.com/")
    gcode.append("M3S250 ; raise")
    gcode.append(f"G1 X0 Y0 {feed_rate_lines} ; Start at center")

    for i, (hA_1, hB_0) in enumerate(line_list):
        hA_0 = hA_1 + 1 if hA_1 % 2 == 0 else hA_1 - 1
        hB_1 = hB_0 + 1 if hB_0 % 2 == 0 else hB_0 - 1

        if not flip_hook_parity:
            if i == 1:
                gcode.append("M3S0 ; lower")  # lower pen after we go from origin -> starting position
            angle = hook_angles[hA_1]
            x, y = radius * math.cos(angle), radius * math.sin(angle)
            gcode.append(f"G1 X{x:.3f} Y{y:.3f} {feed_rate_lines} ; next point")
            continue

        # Notation:
        #   - hA and hB are the start and end hooks
        #   - _0 and _1 are our entry and exit sides for those hooks
        # Assume we're currently at hA_1, having just looped around hA, then the steps are:
        #   (1) Move inside circle, raise pen
        #   (2) Move to next hook (i.e. hB_0), then lower pen
        #   (3) Move outside circle (to perimeter)
        #   (4) Loop around, to outside of hB_1#

        using_arcs = outer_radius == 0

        # (1)
        r = radius - inner_radius
        angle = hook_angles[hA_1] + 0.5 * (hook_angles[hA_1] - hook_angles[hA_0])
        x, y = r * math.cos(angle), r * math.sin(angle)
        gcode.append(f"G1 X{x:.3f} Y{y:.3f} {feed_rate_lines if (i == 0 and use_origin) else feed_rate_arcs} ; move in")
        gcode.append("M3S250 ; raise")

        # Optional debugging: go to the center before moving to the next hook
        if use_origin:
            gcode.append(f"G1 X0 Y0 {feed_rate_lines} ; move to origin")

        # (2)
        angle = hook_angles[hB_0] + 0.5 * (hook_angles[hB_0] - hook_angles[hB_1])
        x, y = r * math.cos(angle), r * math.sin(angle)
        gcode.append(f"G1 X{x:.3f} Y{y:.3f} {feed_rate_lines} ; move to next hook")
        gcode.append("M3S0 ; lower")

        # (3)
        r = radius
        x, y = r * math.cos(angle), r * math.sin(angle)
        gcode.append(f"G1 X{x:.3f} Y{y:.3f} {feed_rate_lines} ; move to perimeter")

        # (4)
        if using_arcs:
            angle_end = hook_angles[hB_1] + 0.5 * (hook_angles[hB_1] - hook_angles[hB_0])
            x_end, y_end = r * math.cos(angle_end), r * math.sin(angle_end)
            angle_mid = 0.5 * (angle + angle_end)
            x_mid, y_mid = r * math.cos(angle_mid), r * math.sin(angle_mid)
            i, j = x_mid - x, y_mid - y
            instruction = "G03" if hB_0 % 2 == 0 else "G02"
            gcode.append(f"{instruction} X{x_end:.3f} Y{y_end:.3f} I{i:.3f} J{j:.3f} {feed_rate_arcs} ; loop around")
        else:
            r = radius + outer_radius
            x, y = r * math.cos(angle), r * math.sin(angle)
            gcode.append(f"G1 X{x:.3f} Y{y:.3f} {feed_rate_arcs} ; move to outside circle")
            angle = hook_angles[hB_1] + 0.5 * (hook_angles[hB_1] - hook_angles[hB_0])
            x, y = r * math.cos(angle), r * math.sin(angle)
            gcode.append(f"G1 X{x:.3f} Y{y:.3f} {feed_rate_arcs} ; loop around")
            r = radius
            x, y = r * math.cos(angle), r * math.sin(angle)
            gcode.append(f"G1 X{x:.3f} Y{y:.3f} {feed_rate_arcs} ; move to inside circle")

    # End at the origin, after raising
    gcode.append("M3S250 ; raise")
    gcode.append(f"G1 X0 Y0 {feed_rate_lines} ; Return to origin to finish")

    return gcode


def format_node(node_idx: int, marked: bool = False) -> tuple[str, str]:
    """
    Takes a node (between 0 and n_nodes) and returns instructions (i.e. side, node identifier, and parity).

    If marked=True then we put a caret before the parity number (this indicates whether the lines we're being shown are
    actually from the prev group, not the current group).
    """
    node = int(node_idx / 2)
    str_node = f"{node // 10: 3} {node % 10}"

    parity = node_idx % 2
    str_parity = ("^" if marked else " ") + str(parity)

    return str_node, str_parity


def generate_instructions_pdf(
    img: Img,
    line_dict: list[tuple[int, int]],
    font_size: int = 32,
    num_cols: int = 3,
    num_rows: int = 20,
    true_x: float = 0.58,
    save_dir: Path | None = None,
    filename: str | None = None,
    show_stats: bool = True,
    true_thread_diameter=0.25,
    font: str = "source-code-pro.ttf",
):
    img_name = img.args.name.split("/")[-1]
    save_dir = save_dir or img.save_dir
    filename = (filename or str(save_dir / f"lines-{img_name}"),)

    try:
        font_file = ROOT_PATH / "lines" / font
        font_name = font.split(".")[0].replace("-", " ").title()
        assert font_file.exists()
        prime_font = TTFont(font_name, str(font_file))
        pdfmetrics.registerFont(prime_font)
    except:
        font_name = ""

    # A4 = (596, 870)
    width = A4[0]
    height = A4[1]

    # Store instructions, in printable form
    lines = []

    total_nlines_so_far = 0
    total_nlines = sum([len(value) for value in line_dict.values()])

    if img.args.group_orders[0].isdigit():
        group_orders = img.args.group_orders
    else:
        d = {color_name[0]: str(idx) for idx, (color_name, color_value) in enumerate(img.palette.items())}
        group_orders = "".join([d[char] for char in img.args.group_orders])

    idx = 0
    curr_color = "not_a_color"
    color_count = {color: [0, 0] for color in set(group_orders)}
    group_len_list = []
    while idx < len(group_orders):
        if group_orders[idx] == curr_color:
            group_len_list[-1][1] += 1
        else:
            group_len_list.append([group_orders[idx], 1])
            curr_color = group_orders[idx][0]
            color_count[group_orders[idx]][1] += 1
        idx += 1

    # group_len_list looks like [['0', 3], ['3', 1], ['1', 2], ... for ███ █ ██ ...
    for group_idx, (color_idx, group_len) in enumerate(group_len_list):
        # Figure out how many groups of this particular colour there are, and which group we're currently on
        n_groups = sum([j[1] for j in group_len_list if j[0] == color_idx])
        group_start = sum([j[1] for j in group_len_list[:group_idx] if j[0] == color_idx])
        group_end = group_start + group_len

        # We'll have some small group overlap, this helps keep track between different groups of the same color
        group_overlap = 5 if group_start != 0 else 0

        # Get the color and color description, from the first letter of the color description (img.e. color_idx="0" gets (0,0,0), "black")
        color_name = list(img.palette.keys())[int(color_idx)]

        # Get the line range we need to draw
        lines_colorgroup = line_dict[color_name]
        line_range = get_range_of_lines(
            n_lines=len(lines_colorgroup),
            n_groups=n_groups,
            group_number=(group_start, group_end),
            h1_offset=group_overlap,
        )

        # Add the start titles to the instructions
        color_count[color_idx][0] += 1
        thiscolor_group = f"{color_count[color_idx][0]}/{color_count[color_idx][1]}"
        lines += [
            "=================",
            f"ByNow = {total_nlines_so_far}/{total_nlines}",
            f"ByEnd = {total_nlines_so_far + len(line_range)}/{total_nlines}",
            f"NOW   = {color_name} {thiscolor_group}",
            "=================",
        ]

        # If this is the first group of the color, then we should start by adding the very first one
        if group_start == 0:
            lines.append(format_node(lines_colorgroup[-1][-1]))

        # Get the points where we'll add in-group progress markers: min of (every 100 lines) and (5 times total)
        progress_gap = min(100, len(line_range) // 5)

        # Add all the instruction lines
        for raw_idx, line_idx in enumerate(line_range):
            # We mark them with a caret if it's in the first 5 nodes of a group which isn't the very first one
            marked = (raw_idx < group_overlap) and (group_start != 0)
            lines.append(format_node(lines_colorgroup[-line_idx][0], marked=marked))
            # We also add progress markers, every 100 lines (or more frequently)
            if (raw_idx % progress_gap == 0) and (0 < raw_idx < len(line_range) - 1):
                lines.append(f"LINES: {raw_idx}/{len(line_range)}")

        # Add the end titles to the instructions
        lines += ["=================", f"DONE  = {color_name} {thiscolor_group}"]
        total_nlines_so_far += len(line_range)

    group_page_list = []
    page_counter = 0

    while len(lines) > 0:
        next_lines, lines = lines[: num_rows * num_cols], lines[num_rows * num_cols :]

        filepath = save_dir / f"lines-{img.args.name.split('/')[-1]}-{page_counter}.pdf"
        canvas = Canvas(str(filepath), pagesize=A4)

        page_counter += 1

        canvas.setLineWidth(0.4)
        canvas.setStrokeGray(0.8)
        canvas.setStrokeColor(gray)

        for col_no, row_no in product(range(num_cols), range(num_rows)):
            x = 0.5 * cm + col_no * (width / num_cols)
            y = height * (1 - (1 + row_no) / (0.6 + num_rows))
            if next_lines:
                next_line = next_lines.pop(0)
            else:
                break

            if isinstance(next_line, str):
                # Case: next line is text that goes between instruction groups
                to = canvas.beginText()
                if font_name != "":
                    to.setFont(font_name, 14 if num_cols == 3 else 18)

                to.setTextOrigin(x, y)
                to.setFillColor(black)
                to.textLine(next_line)

                canvas.drawText(to)

                if next_line.startswith("NOW"):
                    group_page_list.append([page_counter, next_line[6:]])
            else:
                # Case 2: next line is an instruction
                tens, units = next_line

                to = canvas.beginText()
                if font_name != "":
                    to.setFont(font_name, font_size)  # "Symbola"

                to.setTextOrigin(x, y)
                to.setFillColor(black)
                to.textLine(tens)

                to.setTextOrigin(x + 2.45 * (font_size / 20) * cm, y)
                to.setFillColor(blue)
                to.textLine(units)

                canvas.drawText(to)

                canvas_path = canvas.beginPath()
                canvas_path.moveTo(x, y - 0.25 * (font_size / 20) * cm)
                canvas_path.lineTo(x + 3.65 * (font_size / 20) * cm, y - 0.25 * (font_size / 20) * cm)
                canvas.drawPath(canvas_path, fill=0, stroke=1)

        canvas.setLineWidth(3)
        canvas.setStrokeGray(0.5)
        canvas.setStrokeColor(black)
        canvas_path = canvas.beginPath()
        canvas_path.moveTo(width / num_cols, 0)
        canvas_path.lineTo(width / num_cols, height)
        canvas_path.moveTo(2 * width / num_cols, 0)
        canvas_path.lineTo(2 * width / num_cols, height)

        canvas.drawPath(canvas_path, fill=0, stroke=1)

        canvas.save()

    merger = PdfMerger()
    for g in range(page_counter):
        page_filename = save_dir / f"lines-{img.args.name.split('/')[-1]}-{g}.pdf"
        with open(page_filename, "rb") as f:
            merger.append(PdfReader(f))
        os.remove(page_filename)
    for idx, (pagenum, desc) in enumerate(group_page_list):
        merger.add_outline_item(title=f"{idx + 1}/{len(group_page_list)} {desc}", pagenum=pagenum - 1)

    pdf_filename = str(save_dir / f"{filename}.pdf")
    merger.write(pdf_filename)
    print(f"Wrote to {pdf_filename!r}")

    if show_stats:
        df_dicts = {}
        for color_name, lines in line_dict.items():
            nodes = [i[0] // 2 for i in lines]
            counter = Counter(nodes)
            node_frequencies = np.array([counter.get(i, 0) for i in range(max(img.args.d_coords) // 2)])
            node_frequencies_averaged = np.convolve(node_frequencies, np.ones(4), "valid") / 4
            df_dicts[color_name] = node_frequencies_averaged

        fig = px.line(
            pd.DataFrame(df_dicts),
            labels={"index": "node-pair", "value": "# lines", "variable": "Color"},
        )
        fig.update_layout(
            template="ggplot2",
            width=800,
            height=450,
            margin=dict(t=60, r=30, l=30, b=40),
            title_text="Frequencies of lines at nodes, by color",
        )
        fig.show()

        max_colorname_len = max(len(color_name) for color_name in line_dict.keys())
        image_x = max(coord[1] for coord in img.args.d_coords.values())
        image_y = max(coord[0] for coord in img.args.d_coords.values())
        sf = true_x / image_x
        true_y = image_y * sf
        for color_name, lines in line_dict.items():
            total_distance = sum(
                [
                    get_distance(
                        p0=img.args.d_coords[line[0]],
                        p1=img.args.d_coords[line[1]],
                    )
                    for line in lines
                ]
            )
            print(f"{color_name:{max_colorname_len}} | {total_distance * sf / 1000:.2f} km")

        n_buckets = 200

        max_len = 0
        for lines in line_dict.values():
            for line in lines:
                line_len = int(img.args.d_pixels[tuple(sorted(line))].shape[1])
                max_len = max(max_len, line_len)

        true_area_covered_by_thread = 0
        df_dicts = {}
        for color_name, lines in line_dict.items():
            df_dicts[color_name] = [0 for i in range(n_buckets + 1)]
            for line in lines:
                line_len = get_distance(
                    p0=img.args.d_coords[line[0]],
                    p1=img.args.d_coords[line[1]],
                )
                true_area_covered_by_thread += (line_len * (sf * 1000)) * true_thread_diameter
                line_len_bucketed = int(line_len * n_buckets / max_len)
                df_dicts[color_name][line_len_bucketed] += 1

        df_from_dict = pd.DataFrame(df_dicts, index=np.arange(n_buckets + 1) * 100 / n_buckets)

        fig = px.line(
            df_from_dict,
            labels={
                "index": "distance (as % of max)",
                "value": "# lines",
                "variable": "Color",
            },
        )
        fig.update_layout(
            template="ggplot2",
            width=800,
            height=450,
            margin=dict(t=60, r=30, l=30, b=40),
            title_text="Distance of lines, by color",
        )

        fig.add_vline(x=100 * min(image_x, image_y) / max_len, line_width=3)
        if img.args.shape == "Rectangle":
            fig.add_vline(x=100 * max(image_x, image_y) / max_len, line_width=3)
            fig.add_trace(
                go.Scatter(
                    x=[
                        100 * min(image_x, image_y) / max_len + 3,
                        100 * max(image_x, image_y) / max_len + 3,
                    ],
                    y=[
                        0.9 * df_from_dict.values.max(),
                        0.9 * df_from_dict.values.max(),
                    ],
                    text=["x", "y"] if image_x < image_y else ["y", "x"],
                    mode="text",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[100 * image_x / max_len + 3],
                    y=[0.9 * df_from_dict.values.max()],
                    text=["r"],
                    mode="text",
                )
            )
        fig["data"][-1]["showlegend"] = False  # type: ignore
        fig.show()

        true_area = 10e6 * true_x * true_y
        if img.args.shape == "Ellipse":
            true_area *= math.pi / 4
        print(f"""Total area covered (counting overlaps) = {true_area_covered_by_thread / true_area:.3f}

Baseline:
        
MDF: around 0.2 is enough. The stag_large sent to Ben was 0.209. This is with true diameter = 0.25, width 120cm, 10000 lines total. And I think down to 0.2 wouldn't have fucked it up, but smaller values might have.
        
Wheel: 0.264 is what I used for stag, probably 0.3 would have been better (it's got a transparent background!). This would mean about 7200 threads for a standard wheel.
""")
