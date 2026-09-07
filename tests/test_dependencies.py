import os
import subprocess
import sys


def test_import_anacal_does_not_import_galsim():
    """GalSim is not a dependency; a fresh interpreter must not load it.

    Run in a fresh subprocess so that whatever this pytest process has
    already imported cannot mask an import inside anacal.
    """
    code = (
        "import sys, anacal; "
        "assert 'galsim' not in sys.modules, 'anacal imported galsim'; "
        "assert not hasattr(anacal, 'simulation')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
