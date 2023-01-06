import pathlib
from typing import NamedTuple, Optional, Type

import numpy as np
import numpy.typing as npt
from typing_extensions import Self


class YOLOData(NamedTuple):
    num_bb: int
    classes: npt.NDArray[np.int32]
    positions: npt.NDArray[np.float32]
    sizes: npt.NDArray[np.float32]
    confidences: Optional[npt.NDArray[np.float32]] = None

    @classmethod
    def from_txt_file(cls: Type[Self], file_path: pathlib.Path) -> Self:
        with file_path.open("r") as file:
            data = np.loadtxt(file, dtype=np.float32, delimiter=" ")

        assert data.ndim == 2

        num_bb, ncols = data.shape
        assert ncols in [5, 6]

        classes = data[:, 0].astype(np.int32)
        positions = data[:, 1:3]
        sizes = data[:, 3:5]

        if ncols == 5:
            return cls(num_bb, classes, positions, sizes)

        confidences = data[:, 6]

        return cls(num_bb, classes, positions, sizes, confidences)

    @classmethod
    def from_bound_pts(
        cls: Type[Self],
        class_list: list[int],
        pts_list: list[npt.NDArray[np.float32]],
        img_size: tuple[int, int],
    ) -> Self:

        num_bb = len(class_list)
        assert len(pts_list) == num_bb

        classes = np.array(class_list, dtype=np.int32)
        positions = np.empty((num_bb, 2), dtype=np.float32)
        sizes = np.empty((num_bb, 2), dtype=np.float32)

        for i in range(num_bb):
            pos_max = np.max(pts_list[i], axis=0)
            pos_min = np.min(pts_list[i], axis=0)

            pos = (pos_max + pos_min) / 2
            size = pos_max - pos_min + 1

            positions[i] = pos / img_size
            sizes[i] = size / img_size

        return cls(num_bb, classes, positions, sizes)

    def __str__(self) -> str:
        if self.confidences is None:
            return (f"num_bb:{self.num_bb}\n"
                    f"classes:\n{self.classes}\n"
                    f"positions:\n{self.positions}\n"
                    f"sizes:\n{self.sizes}")
        else:
            return (f"num_bb:{self.num_bb}\n"
                    f"classes:\n{self.classes}\n"
                    f"positions:\n{self.positions}\n"
                    f"sizes:\n{self.sizes}\n"
                    f"confidences\n{self.confidences}")

    def to_txt_file(self: Self, file_path: pathlib.Path) -> None:
        with file_path.open("w") as file:
            if self.confidences is None:
                for i in range(self.num_bb):
                    print(self.classes[i], *self.positions[i], *self.sizes[i], file=file)
            else:
                for i in range(self.num_bb):
                    print(self.classes[i], *self.positions[i], *self.sizes[i], self.confidences[i], file=file)

    def get_bboxes(self: Self, img_size: tuple[int, int]) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        half_sizes = self.sizes / 2
        min_positions = self.positions - half_sizes
        max_positions = self.positions + half_sizes
        min_positions = min_positions * img_size
        max_positions = max_positions * img_size
        min_positions = min_positions.astype(np.int32)
        max_positions = max_positions.astype(np.int32)
        return (min_positions, max_positions)
