import torch
import torch.nn.functional as F
import numpy as np


def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)


def torch_to_numpy(a: torch.Tensor):
    return a.squeeze(0).permute(1,2,0).numpy()


def preprocrss(im: torch.Tensor, mean, std):
    im = im / 255
    im = im[:, [2, 1, 0], :, :]
    im -= mean
    im /= std
    return im


def imnormalize(self, img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    if to_rgb:
        img = img[:, :, ::-1]
        # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img - mean) / std


def imdenormalize(self, img, mean, std, to_bgr=False):
    img = ((img * std) + mean)
    if to_bgr:
        img = img[:, :, ::-1]
        # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img