import enum
from itertools import product
from typing import Final, NamedTuple

import cv2
import numpy as np
import numpy.typing as npt

from utils import split_by_cond


@enum.unique
class Direction(enum.IntEnum):
    VERTICAL = enum.auto()
    HORIZONTAL = enum.auto()


# TODO borderを相対座標にするか絶対座標にするか
class WarpBorder(NamedTuple):
    position: float
    top_or_left: float
    bottom_or_right: float
    is_relative: bool


class WarpRegion(NamedTuple):
    vertices: npt.NDArray[np.float32]
    matrix: npt.NDArray[np.float64]

    def __str__(self) -> str:
        return f"vertices:\n{self.vertices}\nmatrix:\n{self.matrix}"


def split_by_border(
    border: WarpBorder,
    quad: npt.NDArray[np.float32],
    direction: Direction,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:

    assert quad.ndim == 2

    assert direction in [e.value for e in Direction]

    if direction == Direction.HORIZONTAL:
        return split_by_cond(quad, quad[:, 0] <= border.position)
    else:
        return split_by_cond(quad, quad[:, 1] <= border.position)


def get_intersect_pts(
    border: WarpBorder,
    quad: npt.NDArray[np.float32],
    direction: Direction,
) -> npt.NDArray[np.float32]:

    assert quad.ndim == 2

    assert direction in [e.value for e in Direction]

    if direction == Direction.HORIZONTAL:
        # borderのx,quadのtopのy,quadのbottomのy
        points = [
            [border.position, quad[0][1]],
            [border.position, quad[2][1]],
        ]
        return np.array(points)
    else:
        # quadのleftのx,rightのx,borderのy
        points = [
            [quad[0][0], border.position],
            [quad[1][0], border.position],
        ]
        return np.array(points)


def borders_to_ctrl_pts(
    borders: list[WarpBorder],
    direction: Direction,
    img_size: tuple[int, int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:

    num_borders = len(borders)
    assert num_borders >= 2

    pts_before: npt.NDArray[np.float32] = np.empty((num_borders, 2, 2), np.float32)
    pts_after: npt.NDArray[np.float32] = np.empty((num_borders, 2, 2), np.float32)

    assert direction in [e.value for e in Direction]

    width, height = img_size
    LENGTH: Final[int] = width if direction == Direction.HORIZONTAL else height

    for i in range(num_borders):
        # before
        pts_before[i, :] = [
            [0, borders[i].position],
            [LENGTH, borders[i].position],
        ]
        # after
        pts_after[i, :] = [
            [borders[i].top_or_left, borders[i].position],
            [borders[i].bottom_or_right, borders[i].position],
        ]

    # 垂直方向なら転置
    if direction == Direction.VERTICAL:
        pts_before = pts_before.transpose((1, 0, 2))
        pts_after = pts_after.transpose((1, 0, 2))

    return (pts_before, pts_after)


def get_mesh_trasnform(points_before: npt.NDArray[np.float32],
                       points_after: npt.NDArray[np.float32]) -> list[list[WarpRegion]]:

    assert points_before.shape == points_after.shape, \
        "The shapes of points_before and points_after do not match: "\
        f"points_before.shape={points_before.shape}, points_after.shape={points_after.shape}"

    assert points_before.ndim == 3, \
        f"ndim of points must be 3, but now {points_before.ndim}"

    num_x, num_y, dim = points_before.shape
    num_x = num_x - 1
    num_y = num_y - 1

    assert dim == 2, f"point dim must be 2, but now {dim}"
    assert (num_x >= 1 and num_y >= 1), "region less"

    regions: list[list[WarpRegion]] = []

    for x in range(num_x):
        regions.append([])

        for y in range(num_y):
            # 四角形領域の頂点の座標取得用のファンシーインデックスのリストを作成する
            fancy_indices = ([x, x, x + 1, x + 1], [y, y + 1, y + 1, y])

            # ファンシーインデックスをもとに変換前と変換後の四角形領域の端点の座標を取得する
            region_before: npt.NDArray[np.float32] = points_before[fancy_indices]
            region_after: npt.NDArray[np.float32] = points_after[fancy_indices]

            # 変換前と変換後の四角形領域の端点の座標をもとに射影変換行列を計算する
            mat: npt.NDArray[np.float64] = cv2.getPerspectiveTransform(region_before, region_after, cv2.DECOMP_QR)

            regions[x].append(WarpRegion(matrix=mat, vertices=region_before))

    return regions


def mesh_transform(
    quads: npt.NDArray[np.float32],
    regions: list[list[WarpRegion]],
    borders: list[WarpBorder],
    direction: Direction,
) -> list[npt.NDArray[np.float32]]:

    assert direction in [e.value for e in Direction]

    NUM_X: Final[int] = len(regions)
    NUM_Y: Final[int] = len(regions[0])
    assert (NUM_X >= 1 and NUM_Y >= 1), "region less"

    NUM_REGION: Final[int] = NUM_X if direction == Direction.HORIZONTAL else NUM_Y
    assert NUM_REGION + 1 == len(borders)

    assert quads.ndim == 3, "ndim"
    num_char, _, _ = quads.shape

    # 頂点の座標をその頂点が属する領域ごとに記録する変数
    pts_per_region: list[list[npt.NDArray[np.float32]]] = [[]]
    # 各頂点が由来する文字のインデックスを<pts_per_region>に合わせて記録する変数
    pts_char_indices: list[list[npt.NDArray[np.int32]]] = [[]]

    def go_next_region(region_idx: int) -> int:
        nonlocal pts_per_region
        nonlocal pts_char_indices

        # 探索位置を次の領域へ進める
        region_idx = region_idx + 1
        assert region_idx < NUM_REGION
        # 次の領域で得た情報の格納先を作る
        pts_per_region.append([])
        pts_char_indices.append([])
        # 探索位置を返す
        return region_idx

    def record_pts(recorded_pts: npt.NDArray[np.float32], char_idx: int, region_idx: int) -> None:
        nonlocal pts_per_region
        nonlocal pts_char_indices
        # 頂点の座標とその頂点が由来する文字のインデックスを記録する
        pts_per_region[region_idx].append(recorded_pts)
        pts_char_indices[region_idx].append(np.full(len(recorded_pts), char_idx, dtype=np.int32))

    region_idx: int = 0
    for char_idx in range(num_char):
        in_pts = np.empty(0, dtype=np.float32)
        notin_pts = np.empty(0, dtype=np.float32)

        # in_ptsが見つかるまでスキップ
        while True:
            (in_pts, notin_pts) = split_by_border(borders[region_idx + 1], quads[char_idx], direction)
            # 見つかったら抜ける
            if len(in_pts) > 0:
                break
            # 見つからない場合は次の領域へ
            region_idx = go_next_region(region_idx)

        # 見つかったin_ptsを登録する
        record_pts(in_pts, char_idx, region_idx)

        # notin_ptsがなくなるまで順に領域を探索
        while len(notin_pts) > 0:
            # 境界との交点を登録
            intersect_pts = get_intersect_pts(borders[region_idx + 1], quads[char_idx], direction)
            record_pts(intersect_pts, char_idx, region_idx)
            # 次の領域へ
            region_idx = go_next_region(region_idx)
            # 領域内で見つかった点を登録し、見つかっていない点があれば次のループへ
            (in_pts, notin_pts) = split_by_border(borders[region_idx + 1], notin_pts, direction)
            record_pts(in_pts, char_idx, region_idx)

    ls_pts: list[npt.NDArray[np.float32]] = [np.concatenate(pts_per_region[i]) for i in range(NUM_REGION)]
    ls_idx: list[npt.NDArray[np.int32]] = [np.concatenate(pts_char_indices[i]) for i in range(NUM_REGION)]

    for i in range(NUM_REGION):
        tmp_pts = np.expand_dims(ls_pts[i], axis=0)

        if direction == Direction.HORIZONTAL:
            tmp_pts = cv2.perspectiveTransform(tmp_pts, regions[i][0].matrix)
        else:
            tmp_pts = cv2.perspectiveTransform(tmp_pts, regions[0][i].matrix)

        ls_pts[i] = np.squeeze(tmp_pts)

    arr_pts: npt.NDArray[np.float32] = np.concatenate(ls_pts)
    arr_idx: npt.NDArray[np.int32] = np.concatenate(ls_idx)

    return [arr_pts[arr_idx == i, :] for i in range(num_char)]


def warp_mesh(src: npt.NDArray[np.uint8], regions: list[list[WarpRegion]], img_size) -> npt.NDArray[np.uint8]:

    num_x = len(regions)
    num_y = len(regions[0])

    assert (num_x >= 1 and num_y >= 1), "region less"

    result = np.zeros_like(src)

    for x, y in product(range(num_x), range(num_y)):

        # マスクを生成する
        mask = np.zeros_like(src)
        cv2.fillConvexPoly(mask, regions[x][y].vertices.astype(np.int32), (255, 255, 255))

        # マスクを適用する
        region = cv2.bitwise_and(src, mask)

        # 変換する
        warped = cv2.warpPerspective(region, regions[x][y].matrix, img_size)

        # 比較明合成する torch.fmaxやfminも使える
        result = np.maximum(result, warped)

    return result
