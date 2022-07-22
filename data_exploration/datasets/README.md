# Datasets for EDF neural nets 

### meaned_over_ocur_bash_phon.h5, meaned_over_ocur_mixed_phon.h5
* 49 electrodes edf; in `meaned` neural responses are meaned over occurrences in train_set_x 
* test ratio: 0.1
* dataset labels: `train_set_x`, `train_set_y`, `test_set_x`, `test_set_y`, `meaned`

For `git-lfs` (.csv files)
* Deleting file from git-lfs
```
git filter-branch --index-filter \
    'git rm -rf --cached --ignore-unmatch [path_to_file]' HEAD
```
