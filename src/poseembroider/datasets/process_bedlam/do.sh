#!/bin/bash

echo "### This is going to be a loooong process... (~35h)"

echo "### PARSING BEDLAM FILES"
# parse .csv files and animation files (validation: 3h20, training: 9-10h)
python datasets/process_bedlam/process_bedlam_step1.py --fps 6 --split training
python datasets/process_bedlam/process_bedlam_step1.py --fps 6 --split validation

# gather data by image (quick < 5 min)
python datasets/process_bedlam/process_bedlam_step2.py --split training
python datasets/process_bedlam/process_bedlam_step2.py --split validation

# select humans in each image (quick < 1 min)
python datasets/process_bedlam/process_bedlam_step3.py --split training
python datasets/process_bedlam/process_bedlam_step3.py --split validation

echo "### FARTHER SAMPLING"
# farther sample data (validation: 30min, training: 13h)
python datasets/process_bedlam/process_bedlam_step4.py --split training --nb_select 50000
python datasets/process_bedlam/process_bedlam_step4.py --split validation --nb_select 20000

echo "### CAPTIONING"
# captionize (validation: 1h, training: 1h30)
python datasets/process_bedlam/process_bedlam_step5.py --split training  --farther_sample 50000
python datasets/process_bedlam/process_bedlam_step5.py --split validation  --farther_sample 10000

echo "### MINING PAIRS"
# select pose pairs (validation: < 5 min, training: ~2h)
python datasets/process_bedlam/process_bedlam_step6.py --split training --farther_sample 50000 --kind 'in'
python datasets/process_bedlam/process_bedlam_step6.py --split training --farther_sample 50000 --kind 'out'
python datasets/process_bedlam/process_bedlam_step6.py --split validation --farther_sample 10000 --kind 'in'
python datasets/process_bedlam/process_bedlam_step6.py --split validation --farther_sample 10000 --kind 'out'

echo "### PAIR CAPTIONING"
# get modifier texts (validation: <1h , training: 2h40)
python datasets/process_bedlam/process_bedlam_step7.py --split training --farther_sample 50000 --kind 'in'
python datasets/process_bedlam/process_bedlam_step7.py --split training --farther_sample 50000 --kind 'out'
python datasets/process_bedlam/process_bedlam_step7.py --split validation --farther_sample 10000 --kind 'in'
python datasets/process_bedlam/process_bedlam_step7.py --split validation --farther_sample 10000 --kind 'out'