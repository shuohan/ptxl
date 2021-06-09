"""Classes to save checkponits and image during training and validation.

"""
import torch
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import namedtuple
from pathlib import Path
from queue import Queue
from threading import Thread
from enum import Enum
from PIL import Image
from torch.nn.functional import interpolate

from .abstract import Observer
from .utils import NamedData


class Saver(Observer):
    """An abstract class to save the training progress.

    Attributes:
        dirname (str or pathlib.Path): The directory to save results.
        step (int): Save every this number of epochs. Do not save if
            :attr:`step` is less or equal to 0.
        save_init (bool): Save before any weight update.

    """
    def __init__(self, dirname, step=0, save_init=False):
        super().__init__()
        self.dirname = Path(dirname)
        self.step = step
        self.save_init = save_init

    def start(self):
        self.dirname.mkdir(parents=True, exist_ok=True)
        if self.save_init:
            self._save()

    def _save(self):
        """Implement save in this function."""
        raise NotImplementedError

    def _needs_to_update(self):
        rule1 = self.contents.counter['epoch'].index1 % self.step == 0
        rule2 = self.contents.counter['epoch'].has_reached_end()
        rule3 = self.contents.counter['batch'].has_reached_end()
        return (rule1 or rule2) and rule3


class ThreadedSaver(Saver):
    """Saves with a thread.

    """
    def __init__(self, dirname, step=0, save_init=False):
        super().__init__(dirname, step=step, save_init=save_init)
        self.queue = Queue()
        self._thread = self._init_thread()

    def _init_thread(self):
        raise NotImplementedError

    def start(self):
        self._thread.start()
        super().start()

    def close(self):
        self.queue.put(None)
        self._thread.join()


class CheckpointSaver(Saver):
    """Saves model periodically.

    Args:
        kwargs (dict): The other stuff to save.

    """
    def __init__(self, dirname, step=0, save_init=False, **kwargs):
        super().__init__(dirname, step=step, save_init=save_init)
        self.kwargs = kwargs

    def _update(self):
        filename = Path(self.dirname, self._get_counter_named_index())
        contents = self._get_contents_to_save()
        torch.save(contents, filename.with_suffix('.pt'))

    def _get_contents_to_save(self):
        return {self._get_counter_name(): self._get_counter_index(),
                'model_state_dict': self.contents.get_model_state_dict(),
                'optim_state_dict': self.contents.get_optim_state_dict(),
                **self.kwargs}

    def _get_counter_named_index(self):
        return self.contents.counter['epoch'].named_index1

    def _get_counter_name(self):
        return self.contents.counter['epoch'].name

    def _get_counter_index(self):
        return self.contents.counter['epoch'].index1


class SaveType(str, Enum):
    """The type of :class:`SaveImage`.

    Attributes:
        NIFTI: Save the image as a nifit file.

    """
    NIFTI = 'nifti'
    PNG = 'png'
    PNG_NORM = 'png_norm'
    PLOT = 'plot'


class ImageType(str, Enum):
    """The type of image to save.

    Attributes:
        IMAGE: Just an image.
        SEG: The image is a segmentation.
        SIGMOID: Apply sigmoid to the probability map.
        SOFTMAX: Apply softmax to the probability map.

    """
    IMAEG = 'image'
    SEG = 'seg'
    SIGMOID = 'sigmoid'
    SOFTMAX = 'softmax'


def create_save_image(save_type, image_type, save_kwargs):
    """Creates an instance of :class:`SaveImage`.

    Args:
        save_type (enum SaveType or str): The type of :class:`SaveImage`.
        image_type (enum ImageType or str): The type of image to save.
        save_kwargs (dict): The other keyword arguments.

    Returns:
        SaveImage: An instance of :class:`SaveImage`.

    """
    save_type = SaveType(save_type)
    image_type = ImageType(image_type)

    if save_type is SaveType.NIFTI:
        save_image = SaveNifti(**save_kwargs)
    elif save_type is SaveType.PNG_NORM:
        save_image = SavePngNorm(**save_kwargs)
    elif save_type is SaveType.PNG:
        save_image = SavePng(**save_kwargs)
    elif save_type is SaveType.PLOT:
        save_image = SavePlot(**save_kwargs)

    if image_type is ImageType.SEG:
        save_image = SaveSeg(save_image)
    elif image_type is ImageType.SIGMOID:
        save_image = SaveSigmoid(save_image)
    elif image_type is ImageType.SOFTMAX:
        save_image = SaveSoftmax(save_image)

    return save_image


class SaveImage:
    """Writes images to disk.

    Attributes:
        zoom (int): Enlarge the image by this factor.

    """
    def __init__(self, zoom=1, **kwargs):
        self.zoom = zoom

    def save(self, filename, image):
        """Saves an image to filename.

        Args:
            filename (str or pathlib.Path): The filename to save.
            image (torch.Tensor): The image to save.

        """
        raise NotImeplementedError

    def _enlarge(self, image):
        if self.zoom > 1:
            image = image[None, ...]
            image = interpolate(image, scale_factor=self.zoom, mode='nearest')
            image = image.squeeze(dim=0)
        return image


class SaveNifti(SaveImage):
    """Writes images as nifti files.

    Attributes:
        affine (numpy.ndarray): The affine matrix
        header (nibabel.Nifti1Header): The nifti header.

    """
    def __init__(self, zoom=1, affine=np.eye(4), header=None, **kwargs):
        super().__init__(zoom)
        self.affine = affine
        self.header = header

    def save(self, filename, image):
        filename = str(filename)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        if not filename.endswith('.nii') and not filename.endswith('.nii.gz'):
            filename = filename + '.nii.gz'
        image = self._enlarge(image).numpy().squeeze()
        obj = nib.Nifti1Image(image, self.affine, self.header)
        obj.to_filename(filename)


class SavePngNorm(SaveImage):
    """Writes 2D images as .png files with intensity normalization.

    """
    def save(self, filename, image):
        filename = str(filename)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        if not filename.endswith('.png'):
            filename = filename + '.png'
        image = self._enlarge(image).numpy().squeeze()
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val) * 255
        elif max_val == 0:
            image = image
        else:
            image = image / max_val * 255
        obj = Image.fromarray(image.astype(np.uint8))
        obj.save(filename)


class SavePng(SaveImage):
    """Assume the image is [0, 1].

    """
    def save(self, filename, image):
        filename = str(filename)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        if not filename.endswith('.png'):
            filename = filename + '.png'
        image = self._enlarge(image).squeeze().numpy() * 255
        obj = Image.fromarray(image.astype(np.uint8))
        obj.save(filename)


class SavePlot(SaveImage):
    """Saves a 1D curve.

    """
    def save(self, filename, image):
        filename = str(filename)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        if not filename.endswith('.png'):
            filename = filename + '.png'
        image = image.squeeze().numpy()
        plt.cla()
        plt.plot(image)
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().savefig(filename)
        plt.close()


class SaveSeg(SaveImage):
    """Saves a segmentation to a file.

    Attributes:
        save_image (SaveImage): The instance to wrap around.

    """
    def __init__(self, save_image):
        self.save_image = save_image

    def save(self, filename, image):
        image = self._convert(image)
        self.save_image.save(filename, image)

    def _convert(self, image):
        """Converts a soft probability map to hard segmentation."""
        image = torch.argmax(image, dim=0, keepdim=False)
        return image


class SaveSigmoid(SaveSeg):
    """Applies sigmoid before saving the probability map.

    """
    def _convert(self, image):
        image = torch.sigmoid(image)
        return image


class SaveSoftmax(SaveSeg):
    """Applies softmax before saving the probability map.

    """
    def _convert_seg(self, image):
        image = torch.nn.functional.softmax(image, dim=0)
        return image


class ImageThread(Thread):
    """Saves images in a thread.

    Attributes:
        save_image (SaveImage): Save images to files.

    """
    def __init__(self, save_image, queue):
        super().__init__()
        self.save_image = save_image
        self.queue = queue

    def run(self):
        while True:
            data = self.queue.get()
            self.queue.task_done()
            if data is None:
                break
            self.save_image.save(data.name, data.data)


class ImageSaver(ThreadedSaver):
    """Saves images.

    Attributes:
        attrs (list[str]): The attribute names of :attr:`subject` to save.
        save_image (SaveImage): Strategy to save the images.
        queue (queue.Queue): The queue to give data to its thread.

    """
    def __init__(self, dirname, save_image, attrs, step=10, save_init=False,
                 use_new_folder=True):
        self.save_image = save_image
        self.attrs = attrs
        self.use_new_folder = use_new_folder
        super().__init__(dirname, step=step, save_init=save_init)

    def _init_thread(self):
        return ImageThread(self.save_image, self.queue)

    def _update(self):
        for aind, attr in enumerate(self.attrs):
            batch = self.contents.get_tensor(attr, 'cpu')
            if batch is None:
                continue
            elif isinstance(batch, NamedData):
                num_samples = batch.data.shape[0]
                for sind, (name, sample) in enumerate(zip(*batch)):
                    fn = self._get_filename(sind, aind, attr, num_samples)
                    fn = '_'.join([fn, name])
                    self.queue.put(NamedData(fn, sample))
            else:
                num_samples = batch.shape[0]
                for sind, sample in enumerate(batch):
                    fn = self._get_filename(sind, aind, attr, num_samples)
                    self.queue.put(NamedData(fn, sample))

    def _get_filename(self, sample_ind, attr_ind, attr, num_samples):
        attr = attr.replace('_', '-')
        sample_temp = 'sample-%%0%dd' % len(str())
        filename = sample_temp % (sample_ind + 1)
        attr_temp = '%%0%dd' % len(str(len(self.attrs)))
        attr_str = attr_temp % (attr_ind + 1)
        filename = '_'.join([filename, attr_temp % attr_ind, attr])
        named_index = self._get_counter_named_index()
        if self.use_new_folder:
            filename = Path(*named_index, filename)
        else:
            filename = '_'.join([*named_index, filename])
        filename = str(Path(self.dirname, filename))
        return filename

    def _get_counter_named_index(self):
        return self.contents.counter.named_index1
