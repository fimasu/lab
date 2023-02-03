import glob
import random

import cv2
import numpy as np
import numpy.typing as npt
import torch.utils.data as data
import torchvision.transforms as T

# Important paths
Decoration_Data = [img for img in glob.glob('Data/Decoration/*.png')]
training_set_path = "Data/TextEffects"


# Functions and Classes
def default_loader(path):
    return cv2.imread(path)


def ColorChange(Img, randint):
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    Img[:, :, 0] = (Img[:, :, 0] + randint) % 181
    return cv2.cvtColor(Img, cv2.COLOR_HSV2BGR)


def RandomColorType1(Character, random1, random2, random3, r1, r2, r3, r4, r5, r6):
    FG = Character[:, :, 0] / 255.
    FG_ = FG**random1
    Result = Character.copy()
    Result[:, :, 0] = r1 * FG_ + (1 - FG_) * r2
    FG_ = FG**random2
    Result[:, :, 1] = r3 * FG_ + (1 - FG_) * r4
    FG_ = FG**random3
    Result[:, :, 2] = r5 * FG_ + (1 - FG_) * r6
    return Result


class NewDataset(data.Dataset):
    def __init__(self, style_img_bgr: npt.NDArray[np.uint8], glyph_img_bgr: npt.NDArray[np.uint8]):
        super(NewDataset, self).__init__()
        self.Blank_1 = glyph_img_bgr
        self.Blank_1 = cv2.resize(self.Blank_1, (320, 320))
        self.Blank_2 = self.Blank_1.copy()

        self.Stylied_1 = style_img_bgr
        self.Stylied_1 = cv2.resize(self.Stylied_1, (320, 320))
        self.Stylied_2 = self.Stylied_1.copy()

        self.loadSize = 288
        self.fineSize = 256
        self.flip = False

        self.training_set = glob.glob(training_set_path + "/*/train")
        self.loader = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def Process(self, img2, size, x1, y1, flip_rand):
        h, w, c = img2.shape

        if (h != size):
            img2 = cv2.resize(img2, (size, size))
            img2 = img2[x1:x1 + self.fineSize, y1:y1 + self.fineSize, :]

        if (self.flip == 1):
            if flip_rand <= 0.25:
                img2 = cv2.flip(img2, 1)
            elif flip_rand <= 0.5:
                img2 = cv2.flip(img2, 0)
            elif flip_rand <= 0.75:
                img2 = cv2.flip(img2, -1)

        return self.loader(img2)

    def __getitem__(self, index):
        flip_rand = random.random()
        random_style = random.random()

        # Original training data
        if random_style < 0.6:

            # Random Type 1
            if random_style < 0.3:
                random1 = random.random()
                random2 = random.random()
                random3 = random.random()
                r1 = random.randint(0, 255)
                r2 = random.randint(0, 255)
                r3 = random.randint(0, 255)
                r4 = random.randint(0, 255)
                r5 = random.randint(0, 255)
                r6 = random.randint(0, 255)

                Blank_1 = self.Blank_1.copy()
                Blank_2 = self.Blank_2.copy()

                Stylied_1 = RandomColorType1(Blank_1, random1, random2, random3, r1, r2, r3, r4, r5, r6)
                Stylied_2 = RandomColorType1(Blank_2, random1, random2, random3, r1, r2, r3, r4, r5, r6)

            # Training Set
            else:
                training_set_size = len(self.training_set)
                style = random.randint(0, training_set_size - 1)
                content1 = random.randint(0, 854)
                content2 = random.randint(0, 854)
                img1 = default_loader((self.training_set[style] + "/%d.png") % (content1))
                img2 = default_loader((self.training_set[style] + "/%d.png") % (content2))
                h, w, c = img1.shape
                Blank_1 = img1[:, :h, :]
                Blank_2 = img2[:, :h, :]
                Stylied_1 = img1[:, h:, :]
                Stylied_2 = img2[:, h:, :]
                randint = random.randint(-180, 180)
                Stylied_1 = ColorChange(Stylied_1, randint)
                Stylied_2 = ColorChange(Stylied_2, randint)

        # Patches randomly cropped from the given image for one-shot fine-tuning
        else:
            Blank_1 = self.Blank_1.copy()
            Blank_2 = self.Blank_2.copy()
            Stylied_1 = self.Stylied_1.copy()
            Stylied_2 = self.Stylied_2.copy()

        # Processing
        size = self.loadSize

        x1 = random.randint(0, size - self.fineSize)
        y1 = random.randint(0, size - self.fineSize)
        x2 = random.randint(0, size - self.fineSize)
        y2 = random.randint(0, size - self.fineSize)

        Data = {}

        Data['Blank_1'] = self.Process(Blank_1, size, x1, y1, flip_rand)
        Data['Blank_2'] = self.Process(Blank_2, size, x2, y2, flip_rand)

        Stylied_1 = cv2.cvtColor(Stylied_1, cv2.COLOR_BGR2RGB)
        Data['Stylied_1'] = self.Process(Stylied_1, size, x1, y1, flip_rand)

        Stylied_2 = cv2.cvtColor(Stylied_2, cv2.COLOR_BGR2RGB)
        Data['Stylied_2'] = self.Process(Stylied_2, size, x2, y2, flip_rand)
        return Data

    def __len__(self):
        return 1000
