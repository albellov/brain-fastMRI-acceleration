# brain-fastMRI


### Dataset
```
https://disk.yandex.ru/d/bVOlw7W9v2mWeQ
```

### Test models
#### Example for x2 acceleration
- Put the models from `model_x2` dir to models dir;
```bash
mv path_to_data_from_ya_disk/model_x2/G_x2.pth brain-fastMRI/models
```
- Unpack the data to data dir;
```bash
tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_source_test.tar.gz brain-fastMRI/data
tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_source_val.tar.gz brain-fastMRI/data
tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_source_train.tar.gz brain-fastMRI/data

tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_sampled_x2_test.tar.gz brain-fastMRI/data
tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_sampled_x2_val.tar.gz brain-fastMRI/data
tar -zxvf path_to_data_from_ya_disk/source_data/ax_t2_single_sampled_x2_train.tar.gz brain-fastMRI/data
```

- Install package.
```bash
cd brain-fastMRI
pip setup.py install
```
