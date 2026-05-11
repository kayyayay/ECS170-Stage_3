''' Concrete IO class for image datasets (MNIST, ORL, CIFAR) '''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
import numpy as np
from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    dataset_name = None  # 'MNIST', 'ORL', or 'CIFAR'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _to_chw(self, image):
        """
        Normalize a single image to shape (C, H, W) as float32 in [0, 1].
        Handles grayscale (H, W), grayscale with channel (H, W, 1),
        and RGB (H, W, 3) inputs. For ORL, RGB channels are identical,
        so we collapse to a single channel.
        """
        img = np.asarray(image)

        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        elif img.ndim == 3:
            if img.shape[-1] in (1, 3) and img.shape[0] not in (1, 3):
                img = np.transpose(img, (2, 0, 1))

            if (self.dataset_name is not None
                    and self.dataset_name.upper() == 'ORL'
                    and img.shape[0] == 3):
                img = img[0:1, :, :]
        else:
            raise ValueError(f'unexpected image shape: {img.shape}')

        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        return img

    def load(self):
        print('loading data...')

        file_path = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(file_path, 'rb') as f:
            raw = pickle.load(f)

        X_train, y_train = [], []
        X_test, y_test = [], []

        for instance in raw['train']:
            X_train.append(self._to_chw(instance['image']))
            y_train.append(int(instance['label']))

        for instance in raw['test']:
            X_test.append(self._to_chw(instance['image']))
            y_test.append(int(instance['label']))

        X_train = np.stack(X_train, axis=0)
        X_test = np.stack(X_test, axis=0)
        y_train = np.asarray(y_train, dtype=np.int64)
        y_test = np.asarray(y_test, dtype=np.int64)

        if self.dataset_name is not None and self.dataset_name.upper() == 'ORL':
            if y_train.min() == 1:
                y_train = y_train - 1
                y_test = y_test - 1

        print(f'  dataset: {self.dataset_name}')
        print(f'  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
        print(f'  X_test  shape: {X_test.shape},  y_test  shape: {y_test.shape}')
        print(f'  num classes: {len(np.unique(y_train))}')

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        }