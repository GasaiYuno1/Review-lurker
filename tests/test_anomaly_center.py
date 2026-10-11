import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lurker.anomaly import center  # noqa: E402


class CenterTests(unittest.TestCase):
    def test_center_removes_common_component(self):
        rng = np.random.default_rng(0)
        E = np.ones((50, 8)) * 3 + 0.1 * rng.normal(size=(50, 8))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        C = center(E)
        self.assertLess(abs(float((C @ C.T)[np.triu_indices(50, 1)].mean())), 0.05)
        np.testing.assert_allclose(np.linalg.norm(C, axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
