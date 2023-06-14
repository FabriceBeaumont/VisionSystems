# Joint transforms for the disparity datasets. Their form is:
# (img_left_transformed, img_right_transformed), disparity_transformed <- Transform((img_left, img_right), disparity)
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class RandomCrop(object):
    """Randomly crop an image pair and the corresponding disparity map.
    
    This is very close to the transform example from:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    The output size can be specified either as an int (square crop) or as a tuple (y, x)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, imgs, disparity):
        img_left, img_right = imgs

        h, w = img_left.shape[-2:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img_left = img_left[:, top: top + new_h, left: left + new_w]
        img_right = img_right[:, top: top + new_h, left: left + new_w]
        disparity = disparity[:, top: top + new_h, left: left + new_w]

        return (img_left, img_right), disparity
    
class Interpolate(object):
    """Interpolate an image pair and the corresponding disparity map.
    
    Note that the disparities are expected to be in pixel coordinates.
    They must be adjusted as well.
    """

    def __init__(self, scale_factor):
        assert isinstance(scale_factor, (float, tuple))
        if isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
            self.ratio = scale_factor
        else:
            assert len(scale_factor) == 2
            self.scale_factor = scale_factor
            _, self.ratio = scale_factor
        self.interpolate = lambda x: F.interpolate(x, scale_factor=self.scale_factor)

    def __call__(self, imgs, disparity):
        img_left, img_right = imgs

        img_left = self.interpolate(img_left.unsqueeze(0)).squeeze(0)
        img_right = self.interpolate(img_right.unsqueeze(0)).squeeze(0)
        disparity = self.interpolate(disparity.unsqueeze(0)).squeeze(0) * self.ratio

        return (img_left, img_right), disparity
    
class DisparityTransform(object):
    """Apply a given image transform to the left and right image pair and optionally to the disparity map.
    
    Note that this class is not suited for handling nondeterministic transformations.
    The apply_to_disparity flag toggles whether the transform should be applied to the disparity map as well.
    """

    def __init__(self, transform, apply_to_disparity=True):
        self.transform = transform
        self.apply_to_disparity = apply_to_disparity

    def __call__(self, imgs, disparity):
        img_left, img_right = imgs

        img_left, img_right = self.transform(img_left), self.transform(img_right)
        if self.apply_to_disparity:
            disparity = self.transform(disparity)
            
        return (img_left, img_right), disparity

class DisparityTransformCompose(object):
    """Apply a list of disparity transforms to the left and right image pair and the disparity map."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, disparity):
        for transform in self.transforms:
            imgs, disparity = transform(imgs, disparity)
        img_left, img_right = imgs
        
        return (img_left, img_right), disparity
    
class MixUp(object):
    """Randomly blend the input image pair and the corresponding disparity map with another image pair and disparity map, respectively.
    
    The other image pair must be drawn from a Dataset class which is expected to provide compatible images and maps in terms of ranges and shapes.
    Slightly adjusted from https://arxiv.org/pdf/1710.09412.pdf
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.len = len(self.dataset)
        self.eps = 1e-12

    def __call__(self, imgs, disparity):
        img_left, img_right = imgs

        rd_index = np.random.randint(self.len)
        (img_left_rd, img_right_rd), disparity_rd = self.dataset.__getitem__(rd_index)
        
        disparity_mask = disparity < self.eps
        disparity_mask_rd = disparity_rd < self.eps

        alpha = np.random.uniform()
        img_left_interpolate = alpha * img_left + (1 - alpha) * img_left_rd
        img_right_interpolate = alpha * img_right + (1 - alpha) * img_right_rd
        disparity_interpolate = alpha * disparity + (1 - alpha) * disparity_rd
        
        disparity_interpolate[disparity_mask] = 0.
        disparity_interpolate[disparity_mask_rd] = 0.

        return (img_left_interpolate, img_right_interpolate), disparity_interpolate

class CutMix(object):
    """Randomly blend the input image pair and the corresponding disparity map with another image pair and disparity map, respectively.
    
    The other image pair must be drawn from a Dataset class which is expected to provide compatible images and maps in terms of ranges and shapes.
    Implemented according to pseudocode from https://arxiv.org/pdf/1905.04899.pdf
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.len = len(self.dataset)
        _, test_image = dataset.__getitem__(0)
        self.W, self.H = test_image.shape[-2:]

    def __call__(self, imgs, disparity):
        img_left, img_right = imgs
        lam = np.random.uniform()
        r_x = np.random.uniform()*self.W
        r_y = np.random.uniform()*self.H
        # the following two lines correct a typo in the paper pseudocode
        r_w = np.sqrt(1-lam)*self.W
        r_h = np.sqrt(1-lam)*self.H

        x_1 = int(np.round(np.max([r_x-r_w/2,0])))
        x_2 = int(np.round(np.min([r_x+r_w/2,self.W])))
        y_1 = int(np.round(np.max([r_y-r_h/2,0])))
        y_2 = int(np.round(np.min([r_y+r_h/2,self.H])))
        # since we apply CutMix element-wise, replace batch shuffle by this
        mix_ind = np.random.randint(self.len)
        (img_left_mix, img_right_mix), disparity_mix = self.dataset.__getitem__(mix_ind)

        # what to do when the sample is an image
        img_l_out, img_r_out, disparity_out = img_left.clone(), img_right.clone(), disparity.clone()
        img_l_out[:, y_1:y_2, x_1:x_2] = img_left_mix[:, y_1:y_2, x_1:x_2]
        img_r_out[:, y_1:y_2, x_1:x_2] = img_right_mix[:, y_1:y_2, x_1:x_2]
        disparity_out[:, y_1:y_2, x_1:x_2] = disparity_mix[:, y_1:y_2, x_1:x_2]

        return (img_l_out, img_r_out), disparity_out