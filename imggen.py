import enum
from pathlib import Path
import random
from typing import Final, NamedTuple
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
import numpy.typing as npt

from transform import (Direction, WarpBorder, borders_to_ctrl_pts, get_mesh_trasnform, save_borders, warp_mesh)


@enum.unique
class ALIGN(enum.IntEnum):
    CENTER = enum.auto()
    TOP_OR_LEFT = enum.auto()
    BOTTOM_OR_RIGHT = enum.auto()


class Character(NamedTuple):
    class_name: int
    images: list[npt.NDArray[np.uint8]]


# def get_bounding_box(src: npt.NDArray[np.uint8], bin_thresh: int = 100) -> tuple[int, int, int, int]:

#     # ソース画像が二次元配列でない場合グレースケールとして扱えないので弾く
#     assert src.ndim == 2, f"ndim of src must be 2, but now {src.ndim}"

#     # バウンディングボックスの計算のためにソース画像をbin_threshを基準に二値化する
#     _, binarized = cv2.threshold(src, bin_thresh, 255, cv2.THRESH_BINARY)

#     # バウンディングボックスを計算して返す
#     return cv2.boundingRect(binarized)

# def place_glyphs(
#     char_list: list[npt.NDArray[np.uint8]],
#     padding: int = 10,
#     spacing: int = 30,
#     direction: Direction = Direction.X,
#     align: ALIGN = ALIGN.CENTER,
# ) -> npt.NDArray[np.uint8]:

#     assert direction in [e.value for e in Direction]
#     assert align in [e.value for e in ALIGN]

#     # 文字画像の数を取得する
#     NUM_CHAR: Final[int] = len(char_list)
#     assert NUM_CHAR > 0, "char_list should not empty"

#     char_size_array: npt.NDArray[np.int32] = np.empty((NUM_CHAR, 2), np.int32)

#     for i in range(NUM_CHAR):
#         # 対象の文字画像のBBを計算する
#         x_min, y_min, w, h = get_bounding_box(char_list[i])
#         # 対象の文字画像をBBで切り抜く
#         char_list[i] = char_list[i][y_min:(y_min + h), x_min:(x_min + w)]
#         # 画像サイズを記録する
#         char_size_array[i] = [w, h]

#     WIDTH_SUM, HEIGHT_SUM = char_size_array.sum(axis=0)
#     WIDTH_MAX, HEIGHT_MAX = char_size_array.max(axis=0)

#     if direction == Direction.X:
#         IMG_WIDTH: int = WIDTH_SUM + spacing * (NUM_CHAR - 1) + padding * 2
#         IMG_HEIGHT: int = HEIGHT_MAX + padding * 2
#     else:
#         IMG_WIDTH: int = WIDTH_MAX + padding * 2
#         IMG_HEIGHT: int = HEIGHT_SUM + spacing * (NUM_CHAR - 1) + padding * 2

#     img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

#     # 座標の基準値をpaddingで初期化する
#     pos_next: int = padding
#     for i in range(NUM_CHAR):
#         # 対象の文字画像のサイズを取得する
#         w, h = char_size_array[i]

#         # X軸方向に並べる場合
#         if direction == Direction.X:
#             # x_minは座標の基準値に等しい
#             x_min = pos_next
#             # y_minをALIGNに合わせて計算する
#             if align == ALIGN.CENTER:
#                 y_min = int((IMG_HEIGHT - h) / 2)
#             elif align == ALIGN.TOP_OR_LEFT:
#                 y_min = padding
#             else:
#                 y_min = IMG_HEIGHT - h - padding

#         # Y軸方向に並べる場合
#         else:
#             # x_minをALIGNに合わせて計算する
#             if align == ALIGN.CENTER:
#                 x_min = int((IMG_WIDTH - w) / 2)
#             elif align == ALIGN.TOP_OR_LEFT:
#                 x_min = padding
#             else:
#                 x_min = IMG_WIDTH - w - padding
#             # y_minは座標の基準値に等しい
#             y_min = pos_next

#         # x_max, y_maxをx_min, y_minから計算する
#         x_max = x_min + w - 1
#         y_max = y_min + h - 1

#         # BBの範囲から結果画像に配置する
#         img[y_min:(y_max + 1), x_min:(x_max + 1)] = char_list[i]

#         # 次の座標の開始値を計算する
#         if direction == Direction.X:
#             pos_next = x_max + spacing + 1
#         else:
#             pos_next = y_max + spacing + 1

#     return img


def get_rand_borders(num_borders: int) -> list[WarpBorder]:
    borders: list[WarpBorder] = []

    for i in range(num_borders):
        pos = i / (num_borders - 1)
        length = random.random() * 0.5
        rate = random.random()
        borders.append(WarpBorder(pos, length * rate, 1 - length * (1 - rate), True))

    return borders


def main():
    # (設定 1/7) 文字列を決めます。
    text = '鑑（かんが）みるぽほ゜。.・,ｇg９90oO'

    # (設定 2/7) 画像の保存フォルダを決めます。
    data_dir = Path(r'F:\project\images')

    # (設定 3/7) フォントファイルを決めます。
    # (M+ FONTS を使わせていただきました)
    # https://mplus-fonts.osdn.jp/about.html
    font_file = r'F:\project\fonts\mplus-1m-regular.ttf'
    font_size = 64
    font = ImageFont.truetype(font=font_file, size=font_size)

    # (設定 4/7) 背景色を『黒 (0, 0, 0)』に決めます。
    # .getbbox() メソッドが、黒を余白とみなしてくれるので、
    # 黒を使います。
    background_color =

    # (設定 5/7) フォントの色を決めます。
    font_color = (255, 255, 255)

    # (設定 6/7) テキストの描画位置を決めます。
    position = (0, 0)

    # (設定 7/7) 画像サイズを決めます。
    # フォントサイズと画像サイズは、だいたいの値です。
    # 文字が画像の中に納まるように、何回か試して決めました。
    image_size = (96, 96)

    # テキストの文字を 1 つずつ取り出して、描画して保存していきます。
    for (n, letter) in enumerate(text):
        # (描画 1/5) 画像 (Image) オブジェクトを作ります。
        im = Image.new(mode='RGB', size=image_size, color= (0, 0, 0))

        # (描画 2/5) 描画 (ImageDraw) オブジェクトを作ります。
        draw = ImageDraw.Draw(im)

        # (描画 3/5) 文字 (letter) を書きます。
        draw.text(xy=position, text=letter, font=font, fill=font_color)

        # (描画 4/5) 画像が 0 でない (色が黒でない) 領域の
        # 座標 (left, upper, right, lower) を取得します。
        bbox = im.getbbox()

        # (描画 5/5) 取得した座標で切り出します。
        im_crop = im.crop(box=bbox)

        # Unicode コードポイントを表す整数を取得します。
        # オード関数 ord() は、Python の組み込み関数です。
        code_point = ord(letter)

        # ファイル名を決めます。
        # (例) '{テキストの中での位置}_{Unicode コードポイント}.png'
        name = f'{n}_{code_point}.png'

        # ファイルパスを決めます。
        file = data_dir.joinpath(name)

        # 保存します。
        im_crop.save(file)
    return


if __name__ == "__main__":
    # ====================
    # params
    # ====================
    input_dir_name = "input"
    out_dir_name = "character"
    NUM_LOOP: Final[int] = 20
    direction = Direction.X

    borders = [
        WarpBorder(0, 0, 1, True),
        WarpBorder(1, 0, 1, True),
    ]

    # borders = [
    #     WarpBorder(0, 0, 1, True),
    #     WarpBorder(1, 0, 1, True),
    # ]

    # borders = [
    #     WarpBorder(0, 0, 1, True),
    #     WarpBorder(0.5, 0, 1, True),
    #     WarpBorder(1, 0, 1, True),
    # ]

    # borders = [
    #     WarpBorder(0, 0, 1, True),
    #     WarpBorder(0.5, 0, 1, True),
    #     WarpBorder(1, 0, 1, True),
    # ]

    # borders = [
    #     WarpBorder(0, 0, 1, True),
    #     WarpBorder(0.5, 0, 1, True),
    #     WarpBorder(1, 0, 1, True),
    # ]
    # ====================

    input_dir = pathlib.Path(f"./{input_dir_name}/")

    chars = []
    for img_path in input_dir.glob("**/*.png"):
        char = cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)
        chars.append(char)

    img = place_glyphs(chars)
    h, w = img.shape

    borders = get_rand_borders(3)
    (pts_before, pts_after) = borders_to_ctrl_pts(borders, (w, h), direction)
    reagions = get_mesh_trasnform(pts_before, pts_after)
    img = warp_mesh(img, reagions, (w, h))
    cv2.imwrite("sample.png", img)
