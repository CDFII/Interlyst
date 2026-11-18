"""
Turns scenarios/designs into Tkinter canvas instructions.
"""
# Imports
from sim.delay_model import ACTIVE, APPROACHES, Design, Scenario, intergreen
from services.config import SETTINGS

# -- METHODS --
def _pt(a: int, u, v, cx, cy):
    if a == 0:
        return (cx + u, cy - v)
    if a == 1:
        return (cx + v, cy + u)
    if a == 2:
        return (cx - u, cy + v)

    return (cx - v, cy - u)

def lane_moves(left_lanes, thru_lanes, flow):
    L, T, R = flow
    lanes = [{"L"} for _ in range(left_lanes)]

    for j in range(thru_lanes):
        m = set()
        if T > 0:
            m.add("T")

        if left_lanes == 0 and j == 0 and L > 0:
            m.add("L")

        if j == thru_lanes - 1 and R > 0:
            m.add("R")

        lanes.append(m)

    return lanes

def arm_geometry(scenario: Scenario, design: Design):
    active = ACTIVE[scenario.layout]
    n_in, n_out, fill = [0] * 4, [0] * 4, [0] * 4

    for a in active:
        opp = (a + 2) % 4
        n_in[a] = design.left_lanes[a] + design.thru_lanes[a]
        n_out[a] = max(1, design.thru_lanes[opp]) if opp in active else 1
        fill[a] = design.left_lanes[opp] if opp in active else 0

    return n_in, n_out, fill

def build_scenario(scenario: Scenario, design: Design, states: dict | None = None, size: int = 560, lane_width: int = 20):
    cx = cy = size / 2
    active = ACTIVE[scenario.layout]
    n_in, n_out, fill = arm_geometry(scenario, design)
    out_w = [(fill[a] + n_out[a]) * lane_width for a in range(4)]

    xs          = [-n_in[0] * lane_width, out_w[0], -out_w[2], n_in[2] * lane_width]
    ys          = [-n_in[1] * lane_width, out_w[1], -out_w[3], n_in[3] * lane_width]
    x_left      = min(x for a, x in zip((0, 0, 2, 2), xs) if a in active)
    x_right     = max(x for a, x in zip((0, 0, 2, 2), xs) if a in active)
    y_top       = min(y for a, y in zip((1, 1, 3, 3), ys) if a in active)
    y_bottom    = max(y for a, y in zip((1, 1, 3, 3), ys) if a in active)
    edge        = size / 2
    v0 = {
        0:  -y_top,
        1:  x_right,
        2:  y_bottom,
        3:  -x_left
    }

    P = []

    def rect(a: int, u0, u1, va, vb, fill: str):
        (x0, y0), (x1, y1) = _pt(a, u0, va, cx, cy), _pt(a, u1, vb, cx, cy)
        P.append({
            "type":     "rect",
            "box":      (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
            "fill":     fill,
            "outline":  None
        })

    def line(a: int, points, fill: str, width: int = 2):
        P.append({
            "type":     "line",
            "points":   [_pt(a, u, v, cx, cy) for u, v in points],
            "fill":     fill,
            "width":    width
        })

    def poly(a: int, points, fill: str):
        P.append({
            "type":     "poly",
            "points":   [_pt(a, u, v, cx, cy) for u, v in points],
            "fill":     fill
        })

    # Draw base terrain
    P.append({
        "type":     "rect",
        "box":      (0, 0, size, size),
        "fill":     SETTINGS.theme_settings.colors["GRASS"],
        "outline":  None
    })
    for a in active:
        rect(a, -n_in[a] * lane_width, out_w[a], v0[a] - 1, edge, SETTINGS.theme_settings.colors["ROAD"])
    P.append({
        "type":     "rect",
        "box":      (cx + x_left, cy + y_top, cx + x_right, cy + y_bottom),
        "fill":     SETTINGS.theme_settings.colors["ROAD"],
        "outline":  None
    })

    # Median cross strip pattern
    for a in active:
        if fill[a] > 0:
            w = fill[a] * lane_width
            rect(a, 0, w, v0[a], edge, SETTINGS.theme_settings.colors["FILLER"])
            v = v0[a]

            while v + w < edge:
                line(a, [(1, v), (w - 1, v + w - 2)], SETTINGS.theme_settings.colors["HATCH"], 1)
                v += 10

    # Lane markings & Zebra crossings
    for a in active:
        sv = v0[a] + (14 if scenario.pedestrian else 0) # Stop-line distance
        if scenario.pedestrian:
            u = -n_in[a] * lane_width
            while u < n_out[a] * lane_width:
                rect(a, u + 1, min(u + 5, n_out[a] * lane_width), v0[a] + 2, v0[a] + 11, SETTINGS.theme_settings.colors["WHITE"])
                u += 10

        line(a, [(0, sv), (0, edge)], SETTINGS.theme_settings.colors["YELLOW"], 2) # Centre line
        if fill[a] > 0:
            line(a, [(fill[a] * lane_width, sv), (fill[a] * lane_width, edge)], SETTINGS.theme_settings.colors["YELLOW"], 2)    # Secondary line, to draw out median
            line(a, [(0, sv), (fill[a] * lane_width, sv)], SETTINGS.theme_settings.colors["YELLOW"], 2)                         # Cap (line)

        # Dotted lines to seperate lanes
        for index in range(1, n_in[a]):
            v = sv + 8
            while v < edge - 4:
                line(a, [(-index * lane_width, v), (-index * lane_width, min(v + 10, edge))], SETTINGS.theme_settings.colors["WHITE"], 1)
                v += 20

        for index in range(1, n_out[a]):
            u = (fill[a] + index) * lane_width
            v = sv + 8
            while v < edge - 4:
                line(a, [(u, v), (u, min(v + 10, edge))], SETTINGS.theme_settings.colors["WHITE"], 1)
                v += 20

        # Stop-line, colored with signal's state for that lane
        nl, nt = design.left_lanes[a], design.thru_lanes[a]
        groups = []
        if nl > 0:
            groups.append(("L", -nl * lane_width + 2, -2))
            groups.append(("TR", -(nl + nt) * lane_width + 2, -nl * lane_width - 2))
        else:
            groups.append(("TR", -nt * lane_width + 2, -2))

        for kind, u0, u1 in groups:
            color = SETTINGS.theme_settings.signal_colors[states[(a, kind)]] if states and (a, kind) in states else SETTINGS.theme_settings.colors["WHITE"]
            line(a, [(u0, sv), (u1, sv)], color, 5)

        # Lane arrows
        for index, moves in enumerate(lane_moves(nl, nt, scenario.flows[a])):
            u, va = -(index + 0.50) * lane_width, sv + 32
            if "T" in moves:
                line(a, [(u, va + 12), (u, va - 2)], SETTINGS.theme_settings.colors["ARROW"], 2)
                poly(a, [(u, va - 9), (u - 4, va - 2), (u + 4, va - 2)], SETTINGS.theme_settings.colors["ARROW"])
            if "L" in moves:
                line(a, [(u, va + 12), (u, va + 4), (u + 3, va + 3)], SETTINGS.theme_settings.colors["ARROW"], 2)
                poly(a, [(u + 10, va + 3), (u + 3, va - 1), (u + 3, va + 7)], SETTINGS.theme_settings.colors["ARROW"])
            if "R" in moves:
                line(a, [(u, va + 12), (u, va + 4), (u - 3, va + 3)], SETTINGS.theme_settings.colors["ARROW"], 2)
                poly(a, [(u - 10, va + 3), (u - 3, va - 1), (u - 3, va + 7)], SETTINGS.theme_settings.colors["ARROW"])

        # Labels for traffic volume, for each incoming direction
        x, y = _pt(a, -n_in[a] * lane_width / 2, edge - 34, cx, cy)
        total = int(scenario.flows[a].sum())
        P.append({
            "type":     "rect",
            "box":      (x - 30, y - 9, x + 30, y + 9),
            "fill":     SETTINGS.theme_settings.colors["TEXT_BG"],
            "outline":  None
        })
        P.append({
            "type":     "text",
            "pos":      (x, y),
            "text":     f"{APPROACHES[a]}: {total}",
            "fill":     "white",
            "size":     12
        })

    return P

# Builds a timeline bar with cursor showing scenario at that time. 
def build_timeline(scenario: Scenario, design: Design, time: float, width: int = 560, height: int = 58):
    lost = intergreen(scenario.speed_kmh)
    n = len(design.greens)
    cycle = sum(design.greens) + n * lost
    scale = (width - 20) / cycle
    x = 10.00
    
    P = [{
        "type":     "rect",
        "box":      (0, 0, width, height),
        "fill":     "white",
        "outline":  None
    }]

    for phase, green in enumerate(design.greens):
        P.append({
            "type":     "rect",
            "box":      (x, 16, x + green * scale, 38),
            "fill":     SETTINGS.theme_settings.phase_group_colors[phase % 4],
            "outline":  None
        })

        # Create label, displaying phase lenght in timeline (if phase label is long enough to fit text)
        if green * scale > 46:
            P.append({
                "type":     "text",
                "pos":      (x + green * scale / 2, 27),
                "text":     f"P{phase + 1} {green:.0f}s",
                "fill":     "white",
                "size":     11
            })

        x += green * scale
        P.append({
            "type":     "rect",
            "box":      (x, 16, x + lost * scale, 38),
            "fill":     SETTINGS.theme_settings.colors["YELLOW"],
            "outline":  None
        })
        x += lost * scale

    cx = 10 + (time % cycle) * scale
    P.append({"type":   "line", "points":   [(cx, 8), (cx, 46)],                                "fill": "#111111",  "width":    2   })
    P.append({"type":   "text", "pos":      (10, 50),               "text": "0s",               "fill": "#444444",  "size":     10  })
    P.append({"type":   "text", "pos":      (width - 22, 50),       "text": f"{cycle:.0f}s",    "fill": "#444444",  "size":     10  })

    return P

def render_png(prims: list, path: str, width: int, height: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), "white")
    dr = ImageDraw.Draw(img)
    fonts = {}

    for p in prims:
        if p["type"] == "rect":
            dr.rectangle(p["box"], fill=p["fill"], outline=p.get("outline"))
        elif p["type"] == "line":
            dr.line(p["points"], fill=p["fill"], width=p["width"])
        elif p["type"] == "poly":
            dr.polygon(p["points"], fill=p["fill"])
        elif p["type"] == "text":
            size = p["size"]
            fonts.setdefault(size, ImageFont.load_default(size=size))
            dr.text(p["pos"], p["text"], fill=p["fill"], font=fonts[size], anchor="mm")

    img.save(path)