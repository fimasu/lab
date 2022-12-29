import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
# import dataset
from transform import (Direction, WarpBorder, get_mesh_trasnform, mesh_transform, warp_mesh, borders_to_ctrl_pts)

# デバイスを設定する
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")

# 前処理
loader = transforms.Compose([
    transforms.Resize(256),  # リサイズする
    transforms.ToTensor()  # Tensor型に変換する
])

# 後処理
unloader = transforms.ToPILImage()


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)  # type: ignore
    return image.to(device, torch.float)


def image_unloader(img_tensor: torch.Tensor):
    # cpu上に表示したい画像を表すTensorをコピーし、元のTensorへの変更を防ぐ
    image = img_tensor.cpu().clone()
    # image = image.squeeze(0)      # remove the fake batch dimension
    image = image.numpy().transpose((1, 2, 0))
    # image = unloader(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image + 1) * 127.5
    return image.astype(np.uint8)


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


borders = [
    WarpBorder(position=0, top_or_left=0, bottom_or_right=200, is_relative=False),
    WarpBorder(position=500, top_or_left=20, bottom_or_right=180, is_relative=False),
    WarpBorder(position=1000, top_or_left=0, bottom_or_right=200, is_relative=False),
]
num_borders = len(borders)

# 境界跨ぎあり
# quads = [
#     [[100, 50], [200, 50], [200, 150], [100, 150]],
#     [[450, 50], [550, 50], [550, 150], [450, 150]],
#     [[800, 50], [900, 50], [900, 150], [800, 150]],
# ]

# 境界跨ぎなし
quads = [
    [[100, 50], [200, 50], [200, 150], [100, 150]],
    [[300, 50], [400, 50], [400, 150], [300, 150]],
    [[600, 50], [700, 50], [700, 150], [600, 150]],
    [[800, 50], [900, 50], [900, 150], [800, 150]],
]

quads = np.array(quads, dtype=np.float32)
assert quads.ndim == 3, "ndim"
num_char, _, _ = quads.shape

(pts_before, pts_after) = borders_to_ctrl_pts(borders, Direction.HORIZONTAL, (800, 200))
reagions = get_mesh_trasnform(pts_before, pts_after)

img = np.zeros((200, 1000, 3), dtype=np.uint8)

img_before = cv2.polylines(img, quads.astype(np.int32), True, (0, 255, 0))
img_before = cv2.line(img, np.array([500, 0], dtype=np.int32), np.array([500, 199], dtype=np.int32), (255, 0, 0))

direction = Direction.HORIZONTAL
pts = mesh_transform(quads, reagions, borders, direction)

for i in range(len(pts)):
    print(pts[i])

# print(bound_quads)
# bound_quads = cv2.perspectiveTransform(bound_quads, mat)
# print(bound_quads)
