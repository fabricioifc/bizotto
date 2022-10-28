import time
import torch
import os
import cv2
import random
import numpy as np
from skimage import io, img_as_float64
from utils import convert_from_color, get_random_pos

EXTS = ['tif', 'tiff', 'jpg', 'png']

# Dataset class

class DatasetIcmbio(torch.utils.data.Dataset):

    def __init__(self, data_files, label_files, window_size, n_channels = 3, cache = True, augmentation=True):
        super(DatasetIcmbio, self).__init__()

        self.data_files = data_files
        self.label_files = label_files
        self.window_size = window_size
        self.n_channels = n_channels
        self.cache = cache
        self.augmentation = augmentation
        
        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}
            
    
    def __len__(self):
        # Default epoch size is 10 000 samples
        # return 10000
        return 10000

    # Return data_files and label_files
    def get_dataset(self):
        return self.data_files, self.label_files
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)
    
    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = 1/255 * np.asarray(io.imread(self.data_files[random_idx])[:,:,:3].transpose((2,0,1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            image_label = io.imread(self.label_files[random_idx], )[:,:,:3]
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(image_label), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]
        
        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))