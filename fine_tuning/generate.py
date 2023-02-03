import argparse

import cv2
import matplotlib.pyplot as plt
import networks
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision.transforms as T

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--outf', default='result/', help='folder to save output images')
    parser.add_argument('--style_name', help='name of the style image')
    parser.add_argument('--style_path', help='path to the input style image')
    parser.add_argument('--glyph_path', help='path to the corresponding glyph of the input style image')
    parser.add_argument('--content_path', help='path to the target content image')
    parser.add_argument('--save_name', default='name to save')

    plt.switch_backend('agg')

    opt = parser.parse_args()

    opt.cuda = (opt.gpu != -1)
    cudnn.benchmark = True
    device = torch.device("cuda:%d" % (opt.gpu) if opt.cuda else "cpu")

    loader = T.Compose([
        T.ToTensor(),
        T.Resize((256, 256), antialias=True),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # ================   Model   ================
    netG = networks.define_G(9, 3).to(device)
    netG.load_state_dict(torch.load('cache/%s_netG.pth' % (opt.style_name), map_location=lambda storage, loc: storage))
    netG.eval()
    for p in netG.parameters():
        p.requires_grad = False

    # ===============   Processing   ===============
    Blank_1 = cv2.imread(opt.glyph_path)
    Blank_2 = cv2.imread(opt.content_path)
    Stylied_1 = cv2.imread(opt.style_path)
    Stylied_1 = cv2.cvtColor(Stylied_1, cv2.COLOR_BGR2RGB)

    Blank_1 = loader(Blank_1).unsqueeze(0).to(device)
    Blank_2 = loader(Blank_2).unsqueeze(0).to(device)
    Stylied_1 = loader(Stylied_1).unsqueeze(0).to(device)

    Stylied_2 = netG(torch.cat([Blank_2, Blank_1, Stylied_1], 1), 256)
    Stylied_2 = Stylied_2.squeeze().add_(1).div(2)
    vutils.save_image(Stylied_2, f"{opt.outf}/{opt.save_name}.png")
    print("saved data")
