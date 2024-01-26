@echo off
title PCA

mkdir projections
mkdir restored_sounds
mkdir denoised_audios
type nul > components.csv

python compute_components.py -d ./learn_audios -n 15 -m pcavarimax
python project.py -d ./project_audios -c ./components.csv -m pcavarimax
python restore_sound.py -d ./projections -c ./components.csv -m pcavarimax
python denoise.py -d ./restored_sounds

pause
