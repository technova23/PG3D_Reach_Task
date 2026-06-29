from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pg3d.utils.arrays import bool_any, bool_info, float_info, frame_to_numpy, to_numpy
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable


def test_jsonable_converts_paths_numpy_and_tuples() -> None:
    value = {
        "path": Path("artifact"),
        "array": np.asarray([1, 2], dtype=np.int64),
        "scalar": np.float32(0.5),
        "tuple": (np.int64(3),),
    }

    assert jsonable(value) == {
        "path": "artifact",
        "array": [1, 2],
        "scalar": 0.5,
        "tuple": [3],
    }


def test_array_helpers_accept_tensor_like_values() -> None:
    info = {
        "success": torch.tensor([True]),
        "distance": torch.tensor([0.125]),
    }

    assert bool_info(info, "success")
    assert float_info(info, "distance", default=1.0) == 0.125
    assert float_info(info, "missing", default=1.0) == 1.0
    assert bool_any(torch.tensor([False, True]))
    np.testing.assert_array_equal(to_numpy(torch.tensor([1, 2])), np.asarray([1, 2]))


def test_frame_to_numpy_unbatches_single_env_rgb_array() -> None:
    frame = torch.zeros((1, 4, 5, 3), dtype=torch.uint8)

    array = frame_to_numpy(frame)

    assert array.shape == (4, 5, 3)
    assert array.dtype == np.uint8


def test_select_device_cpu() -> None:
    assert select_device("cpu").type == "cpu"
