# FastMRI: acceleration MRI procedure using Image-to-Image translation


The Magnetic Resonance Imaging accelerating could be done by taking fewer measurements, it might decrease the MRI time and medical costs. 

The motivation of the project is to make an algorithm, which increases the speed of the MR procedure with maximal saving the quality of the image, using metrics (MAE, PSNR, SSIM (mean absolute error, peak signal-to-noise ratio,  structural similarity)) at different speed-up rates (x2, x4, x8). 
The fastMRI dataset was used for training and evaluation of machine-learning approaches (GAN) to MR image reconstruction. 

An algorithm of fast MRI was create, with high values of a target metric.

### Dataset (already preprocessed)
```
https://disk.yandex.ru/d/bVOlw7W9v2mWeQ
```

### k-space
The k-space is defined by the space covered by the phase and frequency encoding data.
The relationship between k-space data and image data is the Fourier transformation.

![](https://github.com/albellov/brain-fastMRI/blob/master/images/kspace.jpeg)

### Our solution

During the MRI procedure, the entire k-space is filled in, and then an MRI image is obtained. Filling k-space is a long process that can be calmed down as follows: filling 50% of k-space will speed up the procedure for the patient by ~2 times, filling 25% of k-space by ~4 times, filling 12.5% by ~8 times. Acceleration causes severe image distortion, and the higher the acceleration, the greater the distortion.

![](https://github.com/albellov/brain-fastMRI/blob/master/images/x2.jpeg)

![](https://github.com/albellov/brain-fastMRI/blob/master/images/x4.jpeg)

![](https://github.com/albellov/brain-fastMRI/blob/master/images/x8.jpeg)


### Test models
#### Example for x2 acceleration
- Put the models from `model_x2` dir to models dir;

```
bash
mv path_to_data_from_ya_disk/model_x2/G_x2.pth brain-fastMRI/models
```
- Unpack the data to data dir;

```
bash
tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_source_test.tar.gz brain-fastMRI/data
tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_source_val.tar.gz brain-fastMRI/data
tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_source_train.tar.gz brain-fastMRI/data

tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_sampled_x2_test.tar.gz brain-fastMRI/data
tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_sampled_x2_val.tar.gz brain-fastMRI/data
tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_sampled_x2_train.tar.gz brain-fastMRI/data
```

- Install package.

```
bash
cd rain-fastMRI
pip setup.py install
```
