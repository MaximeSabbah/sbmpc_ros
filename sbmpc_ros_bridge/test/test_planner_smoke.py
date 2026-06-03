from __future__ import annotations

import numpy as np
import pytest

from sbmpc_ros_bridge.planner_smoke import parse_vector


def test_parse_vector_accepts_json_and_csv() -> None:
    expected = np.arange(7, dtype=np.float64)

    np.testing.assert_allclose(parse_vector("[0, 1, 2, 3, 4, 5, 6]", name="--q", default=[]), expected)
    np.testing.assert_allclose(parse_vector("0,1,2,3,4,5,6", name="--q", default=[]), expected)


def test_parse_vector_uses_default_and_validates_size() -> None:
    default = np.ones(7, dtype=np.float64)

    np.testing.assert_allclose(parse_vector(None, name="--q", default=default), default)
    with pytest.raises(ValueError, match="7 values"):
        parse_vector("1,2", name="--q", default=default)
