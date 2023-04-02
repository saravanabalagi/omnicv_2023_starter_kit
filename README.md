# OmniCV 2023 Starter Kit

This repository contains the starter kit for Woodscape Motion Segmentation Challenge 2023 being run in conjunction with IEEE Computer Society Conference on Computer Vision and Pattern Recognition OmniCV 2023 Workshop. Full details on both the workshop and the competition are available at the [OmniCV 2023 webpage](https://sites.google.com/view/omnicv2023/challenges/woodscape-challenge).

We provide the starter kit to help contestants get started with the competition rapidly. The starter kit is provided as a reference implementation and is not the only way to approach the problem. Contestants are free to use any other framework or approach to solve the problem.

The starter kit includes a dataloader, a training script, necessary metrics for offline evaluation, an infer script and a submission bundle generator script. The starter kit is based on [PyTorch](https://pytorch.org/).

## Getting Started

This is a [Poetry](https://python-poetry.org/) based project. To get started, install Poetry and run 
```
poetry install   # to install the dependencies
poetry shell     # to enter the virtual environment
```

`pip` or `conda` can also be used to install the dependencies, check the [pyproject.toml](pyproject.toml) file for the list of dependencies.

## Dataset

We provide a mini dataset containing only 4 sample image sets (rgb_images, previous_images, and motion_annotations) per category for convenience. This is to help contestants understand the file structure and formats, and as such, the content of the images and labels in the original dataset may differ.  

Download sample `datasets.zip` from [releases](https://github.com/saravanabalagi/omnicv_2023_starter_kit/releases) and extract it to the root of the repository. This will create a `datasets` folder at the root of the project with 3 directories:

```
datasets
├── parallel_domain     (training images for parallel domain)
├── test_images         (test images on which inference should be done for submission)
└── woodscape           (training images for woodscape)
```

Please download the full Woodscape and Parallel Domain datasets from the competition page and extract them to the corresponding directories and preserve the above structure for other scripts provided in this repository to work properly.

## Training

The start kit includes a [dataloader](dataloader.py) that can combine and load from multiple datasets with similar structure, and a [training script](train.py) with [metrics](metrics.py) for local evaluation. To start training and validation dry run, run the following command:

```sh
python main.py --config configs/train.yaml
```

## Inference and Submission

To generate a submission zip file,
1. run inference on images, save motion mask predictions to `results` folder
2. copy [train.txt](configs/splits/train.txt) to `results` folder
3. zip the `results` folder file (do not include the `results` folder itself, include only the files inside it)

(1) shall be done using [infer.py](infer.py) script, (2) and (3) shall be done using [bundle_results.sh](bundle_results.sh) script as follows:

```sh
python main.py --config configs/infer.yaml    # to generate predictions for test images in results folder
sh ./bundle_results.sh                        # to generate results.zip file for submission
# for files located elsewhere use sh ./bundle_results.sh <results_dir> <train.txt>
```

Finally, submit the `results.zip` file to the competition page.


