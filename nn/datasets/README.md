# Datasets for EDF neural nets 

### 7electrodes_train.h5, 7electrodes_test.h5
49 electrodes edf; means computed within 7 electrode groups (i.e. v_1_x, v_2_x, ...)
dataset labels: 'train_set_x', 'train_set_y', 'test_set_x', 'test_set_y'

For 'git-lfs'
* Deleting file from git-lfs
```
git filter-branch --index-filter \
    'git rm -rf --cached --ignore-unmatch [path_to_file]' HEAD
```
