import argparse
import random
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable, grad

from fine_tuning.dataset import NewDataset
from fine_tuning.networks import define_D, define_G


# ================  Loss  ================
def calc_gradient_penalty(netD, real_data, fake_data, device):

    alpha = torch.rand(real_data.shape[0], 1, 1, 1).to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates,
                     inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                     create_graph=True,
                     retain_graph=True,
                     only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(p=2, dim=1) - 1)**2).mean()  # type: ignore
    return gradient_penalty


def finetune(
    base_net_dir: str,
    style_img_bgr: npt.NDArray[np.uint8],
    glyph_img_bgr: npt.NDArray[np.uint8],
    batch_size: int = 2,
    num_iter: int = 300,
    learning_rate: float = 1e-4,
    gpu_id: int = -1,
    tuned_net_name: str = "effect_netG",
    manual_seed: Optional[int] = None,
):

    if manual_seed is None:
        manual_seed = random.randint(1, 10000)
    print(f"Random Seed: {manual_seed}")
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    if gpu_id != -1:
        torch.cuda.manual_seed_all(manual_seed)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    cudnn.benchmark = True

    # ================  Model  ================
    netG = define_G(9, 3).to(device)
    netD = define_D(12).to(device)

    netG.load_state_dict(torch.load(f"{base_net_dir}/netG.pth", map_location=lambda storage, loc: storage))
    netD.load_state_dict(torch.load(f"{base_net_dir}/netD.pth", map_location=lambda storage, loc: storage))

    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=learning_rate, betas=(0.5, 0.9))
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=learning_rate, betas=(0.5, 0.9))

    criterion = nn.L1Loss()

    # ================  Dataset  ================
    new_dataset = NewDataset(style_img_bgr, glyph_img_bgr)
    loader_ = torch.utils.data.DataLoader(dataset=new_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    loader = iter(loader_)

    # ================  Training  ================
    CRITIC_ITERS = 2
    lambda_gp = 10
    current_size = 256
    Min_loss = 100000

    for iteration in range(1, num_iter + 1):

        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():
            p.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = False

        for i in range(CRITIC_ITERS):
            # 1. generate results of netG
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(loader_)
                data = next(loader)

            Blank_1 = data["Blank_1"].to(device)
            Blank_2 = data["Blank_2"].to(device)
            Stylied_1 = data["Stylied_1"].to(device)
            Stylied_2 = data["Stylied_2"].to(device)

            Stylied_2_recon = netG(torch.cat([Blank_2, Blank_1, Stylied_1], 1), current_size)

            # 2. train netD
            input_real = torch.cat([Blank_2, Blank_1, Stylied_1, Stylied_2], 1)
            input_fake = torch.cat([Blank_2, Blank_1, Stylied_1, Stylied_2_recon], 1)

            netD.zero_grad()
            D_real = netD(input_real).mean()
            D_fake = netD(input_fake).mean()
            gradient_penalty = calc_gradient_penalty(netD, input_real.data, input_fake.data, device)
            errD = D_fake.mean() - D_real.mean() + lambda_gp * gradient_penalty
            errD.backward()
            Wasserstein_D = (D_real.mean() - D_fake.mean()).data.mean()
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = True

        netG.zero_grad()

        # 1. load data
        try:
            data = next(loader)
        except StopIteration:
            loader = iter(loader_)
            data = next(loader)

        Blank_1 = data["Blank_1"].to(device)
        Blank_2 = data["Blank_2"].to(device)
        Stylied_1 = data["Stylied_1"].to(device)
        Stylied_2 = data["Stylied_2"].to(device)

        # 2. netG process
        Stylied_2_recon = netG(torch.cat([Blank_2, Blank_1, Stylied_1], 1), 256)

        # 3. loss
        errS2 = torch.mean(torch.abs(Stylied_2_recon - Stylied_2))

        input_fake = torch.cat([Blank_2, Blank_1, Stylied_1, Stylied_2_recon], 1)
        errD = netD(input_fake).mean()

        G_cost = errS2 * 100 - errD

        # 4. back propogation and update
        G_cost.backward()
        optimizerG.step()

        print("[%d/%d] Loss_L1: %.4f Loss_adv: %.4f Wasserstein_D: %.4f" %
              (iteration, num_iter, errS2.item(), errD.item(), Wasserstein_D.item()))

        if errS2.item() < Min_loss and iteration > 100:
            Min_loss = errS2.item()

            vutils.save_image(Stylied_1.data, "checkpoints/input_style.png", normalize=True)
            vutils.save_image(Stylied_2_recon.data, "checkpoints/output.png", normalize=True)
            vutils.save_image(Stylied_2.data, "checkpoints/ground_truth.png", normalize=True)
            torch.save(netG.state_dict(), f"cache/{tuned_net_name}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=2, help="batch size")
    parser.add_argument("--niter", type=int, default=300, help="number of iterations for fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate, default=0.0002")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu device, -1 for cpu")
    parser.add_argument("--netf", help="where are netG.pth and netD.pth")
    parser.add_argument("--tuned-net-name", help="fine-tuned netG name")
    parser.add_argument("--manualSeed", type=int, help="manual seed")
    parser.add_argument("--style_path", help="path to the style image")
    parser.add_argument("--glyph_path", help="path to the corresponding glyph of the style image")

    plt.switch_backend('agg')
    opt = parser.parse_args()
    print(opt)

    finetune(batch_size=opt.batchSize,
             num_iter=opt.niter,
             learning_rate=opt.lr,
             base_net_dir=opt.netf,
             glyph_img_bgr=cv2.imread(opt.glyph_path),
             style_img_bgr=cv2.imread(opt.style_path),
             gpu_id=opt.gpu,
             manual_seed=opt.manualSeed,
             tuned_net_name=opt.tuned_net_name)
