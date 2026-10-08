"""Detection on real OVITO renders of a nickel bicrystal.

Consecutive renders are not framed identically (the image size changes by a
few dozen pixels), so movement is checked on the boundary's position relative
to the sample's left and right edges as well as in raw pixels.
"""

import numpy as np
import pytest

from conftest import OVITO
from gbtrack.detect import find_boundaries
from gbtrack.track import frame_profile


def middle_boundary(name: str) -> tuple[float, float]:
    """(x in px, x as a fraction of the sample width) of the middle boundary."""
    profile, (left, right) = frame_profile(OVITO / name)
    centre = (left + right) / 2
    middle = min(find_boundaries(profile, n=2), key=lambda x: abs(x - centre))
    return middle, (middle - left) / (right - left)


@pytest.mark.parametrize("temp", ["1000K", "1200K"])
def test_middle_boundary_moves_right(temp: str) -> None:
    x0, f0 = middle_boundary(f"ni_{temp}_step000.png")
    x1, f1 = middle_boundary(f"ni_{temp}_step199.png")
    assert 0.4 < f0 < 0.6
    assert x1 > x0 and f1 > f0


def test_1400K_step0_boundary_found() -> None:
    x0, f0 = middle_boundary("ni_1400K_step000.png")
    assert np.isfinite(x0) and 0.4 < f0 < 0.6


def test_1400K_step199_is_reported_without_asserting_direction() -> None:
    # Thermal disorder at 1400 K turns much of the crystal white; the detector
    # still has to return a position inside the sample.
    x1, f1 = middle_boundary("ni_1400K_step199.png")
    assert np.isfinite(x1) and 0.0 < f1 < 1.0
