# OmniCV 2023 Starter Kit

This repository contains the starter kit for Woodscape Motion Segmentation Challenge 2023 being run in conjunction with IEEE Computer Society Conference on Computer Vision and Pattern Recognition OmniCV 2023 Workshop. Full details on both the workshop and the competition are available at the [OmniCV 2023 webpage](https://sites.google.com/view/omnicv2023/challenges/woodscape-challenge).

## Getting Started

This is a [Poetry](https://python-poetry.org/) based project. To get started, install Poetry and run 
```
poetry install   # to install the dependencies
poetry shell     # to enter the virtual environment
```

`pip` or `conda` can also be used to install the dependencies, check the [pyproject.toml](pyproject.toml) file for the list of dependencies.

## Dataset

Download `datasets.zip` from [releases](https://github.com/saravanabalagi/omnicv_2023_starter_kit/releases) and extract it to the root of the repository. This will create a `datasets` folder at the root of the project with 3 directories:

```
datasets
├── infer_images        (test images for inference)
├── parallel_domain     (training images for parallel domain)
└── woodscape           (training images for woodscape)
```

Please download the full Woodscape and Parallel Domain datasets from the competition page and extract them to the corresponding directories to preserve the above structure.

## Training

The start kit includes a [dataloader](dataloader.py), and a [training script](train.py) with [metrics](metrics.py) for local evaluation. To start training and validation dry run, run the following command:

```sh
python main.py --config configs/train.yaml
```

## Submission

To generate a submission zip file,
1. run inference on images, save motion mask predictions to `results` folder
2. generate `imgs_used.txt` file by combining [train.txt](configs/splits/train.txt) and [val.txt](configs/splits/val.txt) files
3. Zip the `results` folder file

(1) shall be done using [infer.py](infer.py) script, (2) and (3) shall be done using [bundle_results.sh](bundle_results.sh) script as follows:

```sh
python main.py --config configs/infer.yaml    # to generate predictions for test images in results folder
sh ./bundle_results.sh                        # to generate results.zip file for submission
# for files located elsewhere use sh ./bundle_results.sh <results_dir> <train.txt> <val.txt> 
```

Finally, submit the `results.zip` file to the competition page.


