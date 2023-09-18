import unittest
import subprocess


class TestBinExecutables(unittest.TestCase):
    def test_simulate(self):
        subprocess.run(
            ["anacal_simulate_image.py", "--config", "config_sim_lsst.ini",
                "--min_id", "0", "--max_id", "1"],
            capture_output=False,
        )


if __name__ == '__main__':
    unittest.main()
