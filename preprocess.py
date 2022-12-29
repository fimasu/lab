import numpy as np
import numpy.typing as npt
import cv2


def append_distance_info(src: npt.NDArray[np.uint8], bin_thresh: int = 100) -> npt.NDArray[np.uint8]:
    """_summary_

    Args:
        src (npt.NDArray[np.uint8]): source grayscale image
        bin_thresh (int, optional): Threshold for binarizing the source image for distance transformation. Defaults to 100.

    Returns:
        npt.NDArray[np.uint8]: result BGR image
    """

    # ソース画像が二次元配列でない場合グレースケールとして扱えないので弾く
    assert src.ndim == 2, f"ndim of src must be 2, but now {src.ndim}"

    # ソース画像をRチャネルに格納
    result_r = src.copy()

    # 距離変換のためにソース画像をbin_threshを基準に二値化する
    _, binarized = cv2.threshold(result_r, bin_thresh, 255, cv2.THRESH_BINARY)
    binarized_inv = cv2.bitwise_not(binarized)

    # Bチャネルは遠いほど、Gチャネルは近いほど値が大きくなるように距離変換する。
    result_b = cv2.distanceTransform(binarized_inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    result_g = cv2.distanceTransform(binarized, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # 値の範囲を制限する（正規化？）size注意
    result_b = np.clip(result_b, 0, 255).astype(np.uint8)
    result_g = np.clip(result_g, 0, 255).astype(np.uint8)

    # opencvに合わせてBGRの順でチャネル合成する
    return cv2.merge((result_b, result_g, result_r))


path = "zyutu.png"

if __name__ == "__main__":
    # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # print(f"preprocess:{path}")
    # processed = append_distance_info(img)
    # resized = cv2.resize(processed, (128, 128))
    # cv2.imwrite('zyutu_glyph.png', resized)
    # print("done")

    # processed = append_distance_info(img_gray)

    # diff_abs = cv2.absdiff(img_cv2, processed)
    # diff_abs_b, diff_abs_g, diff_abs_r = cv2.split(diff_abs)

    # print(f"max of diff (B, G, R) = ({diff_abs_b.max()}, {diff_abs_g.max()}, {diff_abs_r.max()})")
    # processed = cv2.imread(path)
    # resized = cv2.resize(processed, (128, 128))
    # cv2.imwrite('zyutu.png', resized)
    
    print(cv2.mat)
