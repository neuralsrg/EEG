@echo off
title PCA

mkdir projections
mkdir restored_sounds
type nul > components.csv

python compute_components.py -d ./learn_audios -n 15
python project.py -d ./project_audios -c ./components.csv
python restore_sound.py -d ./projections -c ./components.csv

pause
