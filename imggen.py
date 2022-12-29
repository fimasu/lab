import pathlib
import random
from typing import NamedTuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm

from transform import (Direction, WarpBorder, get_mesh_trasnform, mesh_transform, warp_mesh, borders_to_ctrl_pts)


def imshow(image, title: str | None = None, is_gray: bool = False):
    # 画像を生成
    plt.figure()
    # 画像データを貼り付け
    plt.imshow(image)
    # 必要ならグレースケールで表示する
    if is_gray:
        plt.gray()
    # 軸を非表示
    plt.axis("off")
    # タイトルが与えられていればタイトルを設定する
    if title is not None:
        plt.title(title)
    # 表示
    plt.show()


class Character(NamedTuple):
    class_name: int
    images: list[npt.NDArray[np.uint8]]


def get_bounding_box(src: npt.NDArray[np.uint8], bin_thresh: int = 100) -> tuple[int, int, int, int]:

    # ソース画像が二次元配列でない場合グレースケールとして扱えないので弾く
    assert src.ndim == 2, f"ndim of src must be 2, but now {src.ndim}"

    # バウンディングボックスの計算のためにソース画像をbin_threshを基準に二値化する
    _, binarized = cv2.threshold(src, bin_thresh, 255, cv2.THRESH_BINARY)

    # バウンディングボックスを計算して返す
    return cv2.boundingRect(binarized)


# TODO Y軸のズレをどうするか？ 「っ」などは単にバウンディングボックスで切ってもダメ また、縦方向にも並べたい
def generate_img(char_list: list[npt.NDArray[np.uint8]],
                 padding: int = 10,
                 spacing: int = 30) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:

    # 文字画像の数取得
    num_char = len(char_list)
    assert num_char > 0, "char_list should not empty"

    bounding_quads: npt.NDArray[np.float32] = np.empty((num_char, 4, 2), np.float32)
    char_size_array: npt.NDArray[np.int32] = np.empty((num_char, 2), np.int32)

    for i in range(num_char):
        # BB取得
        x_min, y_min, w, h = get_bounding_box(char_list[i])
        # BBで切り抜き
        char_list[i] = char_list[i][y_min:(y_min + h), x_min:(x_min + w)]
        # 画像サイズを記録する
        char_size_array[i] = [w, h]

    width_sum, height_sum = char_size_array.sum(axis=0)
    width_max, height_max = char_size_array.max(axis=0)
    width_mean, height_mean = char_size_array.mean(axis=0)

    img_width = int(width_sum + padding * 2 + spacing * (num_char - 1))
    img_hight = int(height_max + padding * 2)
    img = np.zeros((img_hight, img_width), np.uint8)

    # 次のx座標の開始値
    x_next: int = padding
    for i in range(num_char):
        w, h = char_size_array[i]
        # 結果画像におけるBBの範囲を計算
        x_min = x_next
        y_min = padding + int((height_mean - h) / 2)
        x_max = x_min + w - 1
        y_max = y_min + h - 1
        # BBの範囲からBBの四頂点の座標を計算
        bounding_quads[i] = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]
        # BBの範囲から結果画像に配置
        img[y_min:(y_max + 1), x_min:(x_max + 1)] = char_list[i]
        # 次のx座標の開始値を計算
        x_next = x_max + spacing + 1

    # 画像と文字のBBを返す
    return img, bounding_quads


def quad_to_yolobb(quads: npt.NDArray[np.float32], img_size):
    pos_max = np.max(quads, axis=1)
    pos_min = np.min(quads, axis=1)

    size = pos_max - pos_min + 1
    pos = (pos_max + pos_min) / 2

    size = size / img_size
    pos = pos / img_size

    # YOLO形式:class x y width hight
    return np.concatenate([pos, size], axis=1)


# ====================
# params
# ====================

borders = [
    WarpBorder(position=0, top_or_left=0, bottom_or_right=1, is_relative=True),
    WarpBorder(position=0.5, top_or_left=0.1, bottom_or_right=0.9, is_relative=True),
    WarpBorder(position=1, top_or_left=0, bottom_or_right=1, is_relative=True),
]

input_dir_name = "input"
out_dir_name = "character"
# gen_setting: dict[str, int] = {"train": 20, "val": 5, "test": 5}
gen_setting: dict[str, int] = {"train": 1}

# ====================

input_dir = pathlib.Path(f"./{input_dir_name}/")

char_dict: dict[str, Character] = {}
last_class_id: int = -1
for img_path in input_dir.glob("**/*.png"):
    char = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    class_name = pathlib.Path(img_path).stem
    if class_name in char_dict:
        # クラス名に対応した画像の配列に格納する
        char_dict[class_name].images.append(char)
    else:
        # 最後に使用したクラスのIDを更新
        last_class_id = last_class_id + 1
        # クラスのIDと画像の配列のペアを格納する
        char_dict[class_name] = Character(last_class_id, [char])

# クラス名のリストを取得
class_names = list(char_dict.keys())
assert len(class_names) > 0

plt.ion()

for mode in gen_setting.keys():
    images_dir = pathlib.Path(f"./{out_dir_name}/images/{mode}")
    labels_dir = pathlib.Path(f"./{out_dir_name}/labels/{mode}")
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
    if not labels_dir.exists():
        labels_dir.mkdir(parents=True)

    for index in tqdm.trange(gen_setting[mode], desc=mode, ncols=80):
        num_char = random.randint(4, 8)
        keys = random.choices(class_names, k=num_char)
        components = [random.choice(char_dict[keys[i]][1]) for i in range(num_char)]
        yoloclass = [char_dict[keys[i]][0] for i in range(num_char)]

        img, bound_quads = generate_img(components)
        h, w = img.shape
        img_size = np.array([w, h], np.float32)

        (pts_before, pts_after) = borders_to_ctrl_pts(borders, Direction.HORIZONTAL, (w, h))
        reagions = get_mesh_trasnform(pts_before, pts_after)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = warp_mesh(img, reagions, (w, h))
        # TODO
        print(bound_quads)
        bound_quads = cv2.perspectiveTransform(bound_quads, reagions[0][0].matrix)
        yolobb = quad_to_yolobb(bound_quads, img_size)

        # save data
        with labels_dir.joinpath(f"{mode}{index}.txt").open("w") as file:
            for i, bounding_box in enumerate(yolobb):
                print(yoloclass[i], *bounding_box, file=file)

        cv2.imwrite(images_dir.joinpath(f"{mode}{index}.png").as_posix(), img)

plt.ioff()
plt.show()
