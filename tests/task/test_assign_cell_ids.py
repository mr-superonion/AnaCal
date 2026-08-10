import anacal
import numpy as np

SCALE = 0.2


def brute_force(x1, x2, cells):
    """Reference implementation of the ownership rule.

    Half-open [x0, x1) containment over the cells in list order (a
    source on a shared inner edge belongs to the right/top cell);
    positions outside every inner region get the nearest cell by
    clipped distance, first-wins on ties.
    """
    out = np.empty(len(x1), dtype=int)
    for k in range(len(x1)):
        best = cells[0].index
        best_d2 = np.inf
        found = False
        for b in cells:
            x0, x1b = b.xmin_in * b.scale, b.xmax_in * b.scale
            y0, y1b = b.ymin_in * b.scale, b.ymax_in * b.scale
            if x0 <= x1[k] < x1b and y0 <= x2[k] < y1b:
                out[k] = b.index
                found = True
                break
            dx = max(x0 - x1[k], 0.0, x1[k] - x1b)
            dy = max(y0 - x2[k], 0.0, x2[k] - y1b)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = b.index
        if not found:
            out[k] = best
    return out


def run_case(x1, x2, cells):
    det = anacal.table.make_catalog_empty(
        np.asarray(x1, float), np.asarray(x2, float)
    )
    got = anacal.task.assign_cell_ids(detection=det, cell_list=cells)
    want = brute_force(np.asarray(x1, float), np.asarray(x2, float), cells)
    np.testing.assert_array_equal(got, want)
    return got


def make_cells(nx=500, ny=500):
    return anacal.geometry.get_cell_list(
        img_nx=nx, img_ny=ny, cell_nx=250, cell_ny=250,
        cell_overlap=80, scale=SCALE,
    )


def test_interior_and_random():
    cells = make_cells()
    rng = np.random.RandomState(42)
    # dense random coverage, including outside the image on every side
    x1 = rng.uniform(-30 * SCALE, 530 * SCALE, 5000)
    x2 = rng.uniform(-30 * SCALE, 530 * SCALE, 5000)
    run_case(x1, x2, cells)


def test_shared_inner_edges():
    cells = make_cells()
    # take the ACTUAL interior inner-region boundaries from the cell
    # geometry (their placement depends on how get_cell_list distributes
    # the overlap); a source exactly on one belongs to the right/top cell
    edges = np.array(sorted({
        b.xmin_in for b in cells if b.xmin_in > 0
    }), dtype=float) * SCALE
    cens = np.array(sorted({
        (b.xmin_in + b.xmax_in) / 2.0 for b in cells
    }), dtype=float) * SCALE
    x1, x2 = [], []
    for e in edges:
        for c in cens:
            x1 += [e, c]
            x2 += [c, e]
    # and the exact inner corners
    for ex in edges:
        for ey in edges:
            x1.append(ex)
            x2.append(ey)
    got = run_case(x1, x2, cells)
    # spot-check the rule itself (not just agreement with brute force):
    # a source exactly on the first interior x-edge, at the bottom row,
    # must belong to the cell whose inner region STARTS at that edge
    # (edge cells' inner regions can start slightly outside the image,
    # so the bottom row is min(ymin_in), not necessarily 0)
    b_by_index = {b.index: b for b in cells}
    b = b_by_index[got[0]]
    ymin_bottom = min(bb.ymin_in for bb in cells)
    assert b.xmin_in * SCALE == edges[0] and b.ymin_in == ymin_bottom


def test_outside_fallback_nearest():
    cells = make_cells()
    # far outside each corner and edge midpoint
    x1 = np.array([-50, 550, -50, 550, 250, 250, -50, 550], float) * SCALE
    x2 = np.array([-50, -50, 550, 550, -50, 550, 250, 250], float) * SCALE
    run_case(x1, x2, cells)


def test_dropped_cell_list():
    cells = make_cells()
    # remove the centre cell: its cell's sources fall back to nearest
    dropped = [b for b in cells if not (b.xmin_in == 170
                                         and b.ymin_in == 170)]
    rng = np.random.RandomState(7)
    x1 = rng.uniform(0.0, 500 * SCALE, 2000)
    x2 = rng.uniform(0.0, 500 * SCALE, 2000)
    run_case(x1, x2, dropped)


def test_single_cell():
    cells = anacal.geometry.get_cell_list(
        img_nx=250, img_ny=250, cell_nx=250, cell_ny=250,
        cell_overlap=0, scale=SCALE,
    )
    assert len(cells) == 1
    x1 = np.array([10.0, 200.0, -20.0, 400.0]) * SCALE
    x2 = np.array([10.0, 200.0, 300.0, -5.0]) * SCALE
    got = run_case(x1, x2, cells)
    assert np.all(got == cells[0].index)
