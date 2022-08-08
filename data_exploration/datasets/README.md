# Datasets for EDF neural nets 

For `git-lfs` (.csv files)
* Deleting file from git-lfs
```
git filter-branch --index-filter \
    'git rm -rf --cached --ignore-unmatch [path_to_file]' HEAD
```
