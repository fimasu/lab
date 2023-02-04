import argparse

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
    netG_path: str,
    glyph_img_bgr: npt.NDArray[np.uint8],
    content_img_bgr: npt.NDArray[np.uint8],
    style_img_bgr: npt.NDArray[np.uint8],
    save_path: str,
    gpu_id: int = -1,
) -> None:

    plt.switch_backend("agg")
    cudnn.benchmark = True
    device = torch.device(f"cuda:{gpu_id}" if gpu_id != -1 else "cpu")

    loader = T.Compose([
        T.ToTensor(),
        T.Resize((256, 256), antialias=True),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # ================   Model   ================
    netG = define_G(9, 3).to(device)
    netG.load_state_dict(torch.load(netG_path, map_location=lambda storage, loc: storage))
    netG.eval()
    for p in netG.parameters():
        p.requires_grad = False

    # ===============   Processing   ===============
    style_img_rgb = cv2.cvtColor(style_img_bgr, cv2.COLOR_BGR2RGB)

    glyph_tensor = loader(glyph_img_bgr).unsqueeze(0).to(device)
    content_tensor = loader(content_img_bgr).unsqueeze(0).to(device)
    Style_tensor = loader(style_img_rgb).unsqueeze(0).to(device)

    Stylied_2 = netG(torch.cat([content_tensor, glyph_tensor, Style_tensor], 1), 256)
    Stylied_2 = Stylied_2.squeeze().add_(1).div(2)
    vutils.save_image(Stylied_2, save_path)
    print(f"The generated image has been saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--netG_path", required=True, help="where are fine-tuned netG.pth")
    parser.add_argument("--style_path", required=True, help="path to the input style image")
    parser.add_argument("--glyph_path", required=True, help="path to the corresponding glyph of the input style image")
    parser.add_argument("--content_path", required=True, help="path to the target content image")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device")
    parser.add_argument("--save_path", default="result/ganarated.png", help="path to save output image")

    opt = parser.parse_args()

    glyph_img_bgr = cv2.imread(opt.glyph_path)
    content_img_bgr = cv2.imread(opt.content_path)
    style_img_bgr = cv2.imread(opt.style_path)

    generate(netG_path=opt.netG_path,
             glyph_img_bgr=glyph_img_bgr,
             content_img_bgr=content_img_bgr,
             style_img_bgr=style_img_bgr,
             save_path=opt.save_path,
             gpu_id=opt.gpu)
