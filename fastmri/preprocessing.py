import os

import pydicom

import numpy as np
from skimage.transform import resize


def load_dicom_scans(path, series=None):
    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    if series:
        slices = [s for s in slices if s.SeriesDescription == series]

    return slices


def fill_to_square(slices, const=0):
    '''
    Filling in null values on the left and right up to square.
    '''

    square_slices = []

    for i in range(len(slices)):

        h, w = slices[i].shape

        assert h >= w

        padding_value = (h-w) // 2

        square_slice = np.pad(slices[i], ((0, 0), (padding_value, padding_value)), 
                              mode='constant', constant_values=const)
        square_slices.append(square_slice)

    return square_slices


def get_numpy_data(scans, resize_shape=None):

    image = [s.pixel_array for s in scans]

    if resize_shape:
        image = fill_to_square(image)
        image = [resize(im, resize_shape, preserve_range=True) for im in image]

    image = np.stack(image)
    image = image.astype(np.int16)

    image[image == -2000] = 0

    return np.array(image, dtype=np.int16)


def dicom_to_3d_array(path_to_dicoms, mri_type, first_slices=None, resize_shape=None):
    slices = load_dicom_scans(path_to_dicoms, mri_type)

    if first_slices:
        assert len(slices) >= first_slices
        slices = slices[:first_slices]

    slices_np = get_numpy_data(slices, resize_shape=resize_shape)

    return slices_np


def scale_MRI(image, low=2, high=98):

    lp, hp = np.percentile(image, (low, high))
    image_scaled = np.clip(image, lp, hp)

    image_scaled -= image_scaled.min()
    image_scaled /= image_scaled.max()
    image_scaled = image_scaled.astype(np.float32)

    return image_scaled


def preprocess_dicom(folders, path_to_save, mri_type, first_slices=None, resize_shape=None, verbose=False):

    for j, folder in enumerate(folders, 1):
        try:
            slices_np = dicom_to_3d_array(folder, mri_type, first_slices=first_slices, 
                                          resize_shape=resize_shape)
        except AssertionError:
            continue

        filename = os.path.split(folder)[1]

        if verbose:
            print(f'{j}/{len(folders)} {filename}')

        for i, slice in enumerate(slices_np, 1):

            slice = scale_MRI(slice)
            slice = slice[np.newaxis, :, :]

            with open(os.path.join(path_to_save, f'{filename}_{i}.npy'), 'wb') as f:
                np.save(f, slice)
