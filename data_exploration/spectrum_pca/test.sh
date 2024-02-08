#! /bin/bash

rm -rf /home/srg/Documents/git/EEG/data_exploration/spectrum_pca/projections
# rm /home/srg/Documents/git/EEG/data_exploration/spectrum_pca/components.csv
rm -rf /home/srg/Documents/git/EEG/data_exploration/spectrum_pca/restored_sounds

# python3 compute_components.py -d /home/srg/Documents/MSU/EEG/sounds -n 15 -m pcavarimax -hop 256
python3 project.py -d /home/srg/Documents/MSU/EEG/sounds/test -c /home/srg/Documents/git/EEG/data_exploration/spectrum_pca/components.csv -m pcavarimax
python3 restore_sound.py -d /home/srg/Documents/git/EEG/data_exploration/spectrum_pca/projections -c /home/srg/Documents/git/EEG/data_exploration/spectrum_pca/components.csv -m pcavarimax -high 350
python3 denoise.py -d /home/srg/Documents/git/EEG/data_exploration/spectrum_pca/restored_sounds -s 1.0