#!/bin/bash
# Stop on error
set -e

# This script is used to generate a submission bundle
# for the results of a challenge.
# Params:
# $1 - results directory
# $2 - train txt file

################################################################

# results directory
results_dir=$1

# default results directory
if [ -z "$results_dir" ]; then
    results_dir="results"
fi

# train and val txt files
train_txt=$2

# default train and val txt files
if [ -z "$train_txt" ]; then
    train_txt="configs/splits/train.txt"
fi

################################################################

# combine train and val txt files
cp $train_txt $results_dir/train.txt

# zip results directory
# do NOT include the directory inside the zip file
cd $results_dir
zip -r ../results.zip *
