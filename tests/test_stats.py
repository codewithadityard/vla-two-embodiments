"""Wilson interval against hand-computed values (docs/IMPLEMENTATION_PLAN.md, M5)."""
import pytest

from minivla.stats import wilson


def test_wilson_known_values():
    assert wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    low, high = wilson(0, 50)
    assert low == 0.0 and high == pytest.approx(0.0713, abs=1e-4)
    assert wilson(50, 50)[1] == 1.0
