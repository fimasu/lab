import random
from pathlib import Path

import cv2
import tqdm

from yolodata import YOLOData

dataset_dir_name = "./character"
save_dir_name = './save_image/'
classes = ["A", "B", "C", "D", "E"]


def plot_one_box(min_pos: tuple[int, int], max_pos: tuple[int, int], image, color: list[int], label=None):
    assert len(color) == 3
    # Plots one bounding box on image img
    t_line = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    cv2.rectangle(image, min_pos, max_pos, color, thickness=t_line, lineType=cv2.LINE_AA)
    # label
    if label is not None:
        t_font = max(t_line - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=t_line / 3, thickness=t_font)[0]
        pos = (min_pos[0] + t_size[0], min_pos[1] - t_size[1] - 3)
        cv2.rectangle(image, min_pos, pos, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image,
                    label, (min_pos[0], min_pos[1] - 2),
                    0,
                    t_line / 3, [225, 255, 255],
                    thickness=t_font,
                    lineType=cv2.LINE_AA)


def draw_box_on_image(
    save_dir: Path,
    img_path: Path,
    label_path: Path,
    class_list: list[str],
    color_list: list[list[int]],
):
    image = cv2.imread(img_path.as_posix())

    assert image.ndim == 3
    height, width, _ = image.shape

    data = YOLOData.from_txt_file(label_path)
    (min_positions, max_positions) = data.get_bboxes((width, height))
    for i in range(data.num_bb):
        class_idx = data.classes[i]
        min_pos = tuple(min_positions[i])
        max_pos = tuple(max_positions[i])
        plot_one_box(min_pos, max_pos, image, color=color_list[class_idx], label=class_list[class_idx])

    # save
    save_file_path = save_dir.joinpath(f"{img_path.stem}.png")
    cv2.imwrite(save_file_path.as_posix(), image)


if __name__ == '__main__':
    img_dir = Path(f"{dataset_dir_name}/images/")
    label_dir = Path(f"{dataset_dir_name}/labels/")
    assert img_dir.is_dir()
    assert label_dir.is_dir()

    save_dir = Path(save_dir_name)
    assert save_dir.is_dir()

    path_list: list[tuple[Path, Path]] = []
    for img_path in img_dir.glob("**/*.png"):
        # 画像に対応するラベルのpathを求める
        label_path = img_path.relative_to(img_dir).with_suffix(".txt")
        label_path = label_dir.joinpath(label_path)
        path_list.append((img_path, label_path))

    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for (img_path, label_path) in tqdm.tqdm(path_list):
        draw_box_on_image(save_dir, img_path, label_path, classes, colors)
