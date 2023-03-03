import numpy as np


MARCHING_SQUARES_TABLE = [
    (False, []),  # 0000
    (False, [((0, 3), (2, 3))]),  # 0001
    (False, [((1, 2), (2, 3))]),  # 0010
    (False, [((0, 3), (1, 2))]),  # 0011
    (False, [((0, 1), (1, 2))]),  # 0100
    (True, ([((0, 1), (1, 2)), ((0, 3), (2, 3))], [((0, 1), (0, 3)), ((1, 2), (3, 2))])),  # 0101
    (False, [((0, 1), (2, 3))]),  # 0110
    (False, [((0, 1), (0, 3))]),  # 0111
    (False, [((0, 1), (0, 3))]),  # 1000
    (False, [((0, 1), (2, 3))]),  # 1001
    (True, ([((0, 1), (0, 3)), ((1, 2), (3, 2))], [((0, 1), (1, 2)), ((0, 3), (2, 3))])),  # 1010
    (False, [((0, 1), (1, 2))]),  # 1011
    (False, [((0, 3), (1, 2))]),  # 1100
    (False, [((1, 2), (2, 3))]),  # 1101
    (False, [((0, 3), (2, 3))]),  # 1110
    (False, []),  # 1111
]


def values_to_index(values):
    n = 0
    for v in values:
        if v > 0:
            n += 1
        n = n << 1
    return n >> 1


def lerp_point(p0, p1, v0, v1):
    t = v0/(v0-v1)
    if t < 0:
        t = 0
    elif t > 1:
        t = 1
    return p0*(1-t)+t*p1


def draw_zeros(window, func, color, thickness, square_size, rect_start=None, rect_size=None):
    if rect_start is None:
        rect_start = window.top_left
    if rect_size is None:
        rect_size = window.size
    grid_shape = np.ceil(rect_size/square_size).astype(int)+1
    samples = np.zeros(grid_shape, dtype=float)
    for x, y in np.ndindex(*grid_shape):
        samples[x, y] = func(rect_start+square_size*np.array((x, y), dtype=float))

    for x, y in np.ndindex(*(grid_shape-1)):
        xy = np.array((x, y), dtype=float)
        corners = [
            rect_start+xy*square_size,
            rect_start+(xy+np.array((1, 0), dtype=float))*square_size,
            rect_start+(xy+np.array((1, 1), dtype=float))*square_size,
            rect_start+(xy+np.array((0, 1), dtype=float))*square_size,
        ]
        corner_samples = [samples[x, y], samples[x+1, y], samples[x+1, y+1], samples[x, y+1]]

        sample_center, edges = MARCHING_SQUARES_TABLE[values_to_index(corner_samples)]
        if sample_center:
            center = rect_start+(xy+np.array((.5, .5), dtype=float))*square_size
            edges = edges[int(func(center) > 0)]
        for (i0, i1), (j0, j1) in edges:
            p0 = lerp_point(corners[i0], corners[i1], corner_samples[i0], corner_samples[i1])
            p1 = lerp_point(corners[j0], corners[j1], corner_samples[j0], corner_samples[j1])
            window.draw_line(p0, p1, thickness, color)
