import numpy as np

import anacal


def test_make_catalog_empty_defaults():
    x1 = np.array([12.34, -1.23])
    x2 = np.array([56.78, 4.56])

    sources = anacal.table.make_catalog_empty(x1=x1, x2=x2)

    assert sources.shape == (2,)

    src = sources[0]

    # model centroids and detected positions should match inputs
    assert src.model.x1.v == x1[0]
    assert src.model.x2.v == x2[0]
    assert src.x1_det == x1[0]
    assert src.x2_det == x2[0]
    assert src.block_id == 0

    # All derivative components should be zeroed
    for qn in [
        src.model.F,
        src.model.t,
        src.model.a1,
        src.model.a2,
        src.model.x1,
        src.model.x2,
        src.fluxap2,
        src.fpfs_e1,
        src.fpfs_e2,
        src.fpfs_m0,
        src.fpfs_m2,
        src.peakv,
        src.bkg,
    ]:
        assert qn.g1 == 0.0
        assert qn.g2 == 0.0
        assert qn.x1 == 0.0
        assert qn.x2 == 0.0

    # Ensure weights default to unity with zero derivatives
    assert src.wdet.v == 1.0
    assert src.wsel.v == 1.0
    assert src.wdet.g1 == 0.0
    assert src.wsel.g1 == 0.0
    assert src.wdet.g2 == 0.0
    assert src.wsel.g2 == 0.0
    assert src.wdet.x1 == 0.0
    assert src.wdet.x2 == 0.0
    assert src.wsel.x1 == 0.0
    assert src.wsel.x2 == 0.0

    # Second element should mirror second set of inputs
    other = sources[1]
    assert other.model.x1.v == x1[1]
    assert other.model.x2.v == x2[1]
    assert other.x1_det == x1[1]
    assert other.x2_det == x2[1]

    # Loss bookkeeping should be reset
    assert src.loss.v.v == 0.0
