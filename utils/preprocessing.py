import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


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


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.
    See :class:`~torchvision.transforms.ToPILImage` for more details.
    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    elif isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))

        elif pic.ndimension() == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

    npimg = pic
    if isinstance(pic, torch.Tensor):
        if pic.is_floating_point() and mode != 'F':
            pic = pic.mul(255).byte()
        npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)