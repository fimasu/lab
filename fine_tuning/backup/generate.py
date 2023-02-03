import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import torchvision.utils as vutils

from fine_tuning.networks import define_G


def generate(
    net_dir: str,
    glyph_img_bgr: npt.NDArray[np.uint8],
    content_img_bgr: npt.NDArray[np.uint8],
    style_img_bgr: npt.NDArray[np.uint8],
    save_path: Path,
    gpu_id: int = -1,
) -> None:

    cudnn.benchmark = True
    device = torch.device(f"cuda:{gpu_id}" if gpu_id != -1 else "cpu")

    # GANのジェネレーターを初期化し、state_dictをロードする
    netG = define_G(9, 3).to(device)
    netG_state_dict = torch.load(net_dir, map_location=lambda storage, loc: storage)
    netG.load_state_dict(netG_state_dict)

    # GANのジェネレーターを推論モードに設定する
    netG.eval()
    for p in netG.parameters():
        p.requires_grad = False

    # style画像をRGBに戻す
    stylized_1 = cv2.cvtColor(style_img_bgr, cv2.COLOR_BGR2RGB)

    loader = T.Compose([
        T.ToTensor(),
        T.Resize((256, 256), antialias=True),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    blank_1 = loader(glyph_img_bgr).unsqueeze(0).to(device)  # type: ignore
    blank_2 = loader(content_img_bgr).unsqueeze(0).to(device)  # type: ignore
    stylized_1 = loader(style_img_bgr).unsqueeze(0).to(device)  # type: ignore

    stylized_2: torch.Tensor = netG(torch.cat([blank_2, blank_1, stylized_1], 1), 256)
    stylized_2 = stylized_2.squeeze().add_(1).div(2)

    if not save_path.parent.is_dir():
        save_path.parent.mkdir(parents=True)
    vutils.save_image(stylized_2, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="gpu device")
    parser.add_argument("--style_path", help="path to the input style image")
    parser.add_argument("--glyph_path", help="path to the corresponding glyph of the input style image")
    parser.add_argument("--content_path", help="path to the target content image")
    parser.add_argument("--save_path", default="result/stylized.png", help="path to save output image")
    parser.add_argument('--netf', help='where are fine-tuned netG.pth')
    plt.switch_backend('agg')
    opt = parser.parse_args()

    generate(net_dir=opt.netf,
             glyph_img_bgr=cv2.imread(opt.glyph_path),
             content_img_bgr=cv2.imread(opt.content_path),
             style_img_bgr=cv2.imread(opt.style_path),
             save_path=Path(opt.save_path),
             gpu_id=opt.gpu)
