import os
import subprocess
import sys


def test_import_anacal_needs_only_numpy():
    """numpy is the only dependency; a fresh interpreter must load nothing
    else -- not GalSim, not astropy, not even fitsio (tests only).

    Run in a fresh subprocess so that whatever this pytest process has
    already imported cannot mask an import inside anacal.
    """
    code = (
        "import sys, anacal; "
        "bad = {m.split('.')[0] for m in sys.modules} "
        "& {'galsim', 'astropy', 'fitsio', 'scipy'}; "
        "assert not bad, f'anacal imported {bad}'; "
        "assert not hasattr(anacal, 'simulation')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
