"""Classes to save checkponits and image during training and validation.

"""
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from collections import namedtuple
from pathlib import Path
from queue import Queue
from threading import Thread
from enum import Enum
from PIL import Image
from torch.nn.functional import interpolate

from .observer import Observer
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

    def update_on_train_start(self):
        """Creates the results folder."""
        if self.save_init:
            self._save()

    def _save(self):
        """Implement save in this function."""
        raise NotImplementedError

    def _need_to_save(self):
        """Tests whether need to save at the current epoch."""
        if self.step > 0:
            rule1 = self.subject.epoch_ind % self.step == 0
            rule2 = self.subject.epoch_ind == self.subject.num_epochs
            return rule1 or rule2
        else:
            return False


class ThreadedSaver(Saver):
    """Saves with threads.

    """
    def __init__(self, dirname, step=0, save_init=False):
        super().__init__(dirname, save_init)
        self.queue = Queue()
        self._thread = self._init_thread()

    def _init_thread(self):
        raise NotImplementedError

    def update_on_train_start(self):
        super().update_on_train_start()
        if self.step > 0:
            self._thread.start()

    def update_on_train_end(self):
        if self.step > 0:
            self.queue.put(None)
            self._thread.join()


class CheckpointSaver(Saver):
    """Saves model periodically.

    Attributes:
        kwargs (dict): The other stuff to save.

    """
    def __init__(self, dirname, step=0, save_init=False, **kwargs):
        super().__init__(dirname, save_init)
        self.step = step
        self.kwargs = kwargs

    def update_on_train_start(self):
        pattern = 'epoch-%%0%dd.pt' % len(str(self.subject.num_epochs))
        self._pattern = str(self.dirname.joinpath(pattern))
        super().update_on_train_start()

    def update_on_epoch_end(self):
        """Saves a checkpoint."""
        if self._need_to_save():
            self._save()

    def _save(self):
        self.dirname.mkdir(parents=True, exist_ok=True)
        filename = self._pattern % self.subject.epoch_ind
        contents = {'epoch': self.subject.epoch_ind,
                    'model_state_dict': self.subject.get_model_state_dict(),
                    'optim_state_dict': self.subject.get_optim_state_dict(),
                    **self.kwargs}
        torch.save(contents, filename)


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


def create_save_image(save_type, image_type, zoom=1):
    """Creates an instance of :class:`SaveImage`.

    Args:
        save_type (enum SaveType or str): The type of :class:`SaveImage`.
        image_type (enum ImageType or str): The type of image to save.
        zoom (int): Enlarge the image by this factor.

    Returns:
        SaveImage: An instance of :class:`SaveImage`.

    """
    save_type = SaveType(save_type)
    image_type = ImageType(image_type)

    if save_type is SaveType.NIFTI:
        save_image = SaveNifti(zoom=zoom)
    elif save_type is SaveType.PNG_NORM:
        save_image = SavePngNorm(zoom=zoom)
    elif save_type is SaveType.PNG:
        save_image = SavePng(zoom=zoom)
    elif save_type is SaveType.PLOT:
        save_image = SavePlot()

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
    def __init__(self, zoom=1):
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

    """
    def save(self, filename, image):
        filename = str(filename)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        if not filename.endswith('.nii') and not filename.endswith('.nii.gz'):
            filename = filename + '.nii.gz'
        image = self._enlarge(image).numpy()
        obj = nib.Nifti1Image(image, np.eye(4))
        obj.to_filename(filename)


class SavePngNorm(SaveImage):
    """Writes 2D images as .png files with intensity normalization.

    """
    def save(self, filename, image):
        filename = str(filename)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        if not filename.endswith('.png'):
            filename = filename + '.png'
        image = self._enlarge(image).numpy()
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
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
        image = self._enlarge(image).numpy() * 255
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
        step (int): Save images every this number of epochs.
        queue (queue.Queue): The queue to give data to its thread.
        save_type (str): The type of files to save the images to.
        image_type (str): The type of images to save.
        file_struct (str): Indicates the file structure. For example,
            ``"epoch/batch/sample"`` saves all samples of a mini-batch as
            `dirname/epoch-ind/batch-ind/sample-ind_name.ext``.
            ``"epoch/batch"`` saves the samples as
            ``dirname/epoch-ind/batch-ind_sample-ind_name.ext``.
        zoom (int): Enlarge the image by this factor.
        ordered (bool): Order the saved images according to :attr:`attrs`.
        create_save_image (function): The function to create an instance of
            :class:`SaveImage`.

    """
    def __init__(self, dirname, attrs=[], step=10, save_type='nifti',
                 image_type='image', file_struct='epoch/batch/sample',
                 save_init=False, zoom=1, ordered=False, prefix='',
                 create_save_image=create_save_image):
        self.save_type = save_type
        self.image_type = image_type
        self.attrs = attrs
        self.step = step
        self.file_struct = file_struct
        self.zoom = zoom
        self.ordered = ordered
        self.prefix = prefix
        self.create_save_image = create_save_image
        self._pattern = None
        super().__init__(dirname, save_init=save_init)

    def update_on_train_start(self):
        self._pattern = self._get_filename_pattern()
        super().update_on_train_start()

    def _init_thread(self):
        save = self.create_save_image(self.save_type, self.image_type, self.zoom)
        return ImageThread(save, self.queue)

    def _get_filename_pattern(self):
        epoch_pattern = 'epoch-%%0%dd' % len(str(self.subject.num_epochs))
        batch_pattern = 'batch-%%0%dd' % len(str(self.subject.num_batches))
        sample_pattern = 'sample-%%0%dd' % len(str(self.subject.batch_size))
        all_parts = ['epoch', 'batch', 'sample']
        file_parts = self.file_struct.split('/')
        assert set(file_parts).issubset(set(all_parts))
        pattern = self.dirname
        for ap in all_parts:
            sub_pattern = eval('_'.join([ap, 'pattern']))
            pattern = self._join_pattern(sub_pattern, pattern, ap in file_parts)
        if self.ordered:
            attr_pattern = '%%0%dd' % len(str(len(self.attrs)))
            if len(self.prefix) > 0:
                attr_pattern = '-'.join([self.prefix, attr_pattern])
            pattern = self._join_pattern(attr_pattern, pattern, False)
        pattern = self._join_pattern('%s', pattern, False)
        return pattern

    def _join_pattern(self, sub, all, as_path=True):
        """Concatenates ``sub_pattern`` after ``pattern``."""
        return str(Path(all, sub)) if as_path else '_'.join([all, sub])

    def update_on_batch_end(self):
        if self._need_to_save():
            self._save()

    def _save(self):
        for aind, attr in enumerate(self.attrs):
            batch = getattr(self.subject, attr)
            if isinstance(batch, NamedData):
                for sind, (name, sample) in enumerate(zip(*batch)):
                    filename = self._get_filename(sind + 1, aind + 1, attr)
                    filename = '_'.join([filename, name])
                    self.queue.put(NamedData(filename, sample))
            else:
                for sind, sample in enumerate(batch):
                    filename = self._get_filename(sind + 1, aind + 1, attr)
                    self.queue.put(NamedData(filename, sample))

    def _get_filename(self, sample_ind, attr_ind, attr):
        attr = attr.replace('_', '-')
        if self.ordered:
            return self._pattern % (self.subject.epoch_ind,
                                    self.subject.batch_ind,
                                    sample_ind, attr_ind, attr)
        else:
            return self._pattern % (self.subject.epoch_ind,
                                    self.subject.batch_ind,
                                    sample_ind, attr)
