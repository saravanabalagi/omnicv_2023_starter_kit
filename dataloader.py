from pathlib import Path, PurePath
from PIL import Image
from munch import Munch
from torchvision import transforms
from torch.utils import data
import numpy as np
import torch


class MixedDataset(data.Dataset):
    def __init__(self, root_dir, img_paths_file=None, img_paths_glob=None, infer=False):
        self.infer = infer
        self.root_dir = Path(root_dir)
        self.load_img_paths(img_paths_file, img_paths_glob)
        self.define_transforms()

    def define_transforms(self):
        self.to_tensor_img = transforms.ToTensor()
        self.to_tensor_label_motion = lambda x: torch.LongTensor(np.array(x))

    def load_img_paths(self, img_paths_file, img_paths_glob):
        self.img_paths = []
        if img_paths_file is not None:
            self.img_paths = [line.rstrip('\n') for line in open(img_paths_file)]
        if img_paths_glob is not None:
            self.img_paths = [f.name for f in list(self.root_dir.glob(img_paths_glob))]
        if len(self.img_paths) == 0:
            raise ValueError(f'No images found in {self.root_dir}')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        dataset_record = Munch()

        # define paths to images and labels
        # infer dataset has only image name in the path
        # woodscape dataset has dataset name and image name in the path
        # parallel domain dataset has dataset name, scene name and image name in the path
        img_path = Path(self.img_paths[index])
        img_stem = img_path.stem
        img_ext = img_path.suffix
        img_name = img_stem + img_ext
        img_prev_name = img_stem + '_prev' + img_ext
        img_path_tokens = PurePath(img_path).parts
        dataset_name = img_path_tokens[0] if len(img_path_tokens) > 1 else ""
        scene_name = img_path_tokens[1] if len(img_path_tokens) == 3 else ""
        scene_dir = self.root_dir / dataset_name / scene_name

        # load images
        img_dir = scene_dir / 'rgb_images'
        img_prev_dir = scene_dir / 'previous_images'
        img = Image.open(img_dir / img_name).convert('RGB')
        img_prev = Image.open(img_prev_dir / img_prev_name).convert('RGB')
        img = self.to_tensor_img(img)
        img_prev = self.to_tensor_img(img_prev)

        dataset_record.index = torch.tensor(index)
        dataset_record.img = img
        dataset_record.img_prev = img_prev

        # load labels
        if not self.infer:
            labels_motion_dir = scene_dir / 'motion_annotations' / 'gtLabels'
            # define path to other labels in a similar way if needed, for example:
            # labels_semantic_dir = scene_dir / 'semantic_annotations' / 'gtLabels'
            label_motion = Image.open(labels_motion_dir / img_name).convert('L')
            label_motion = self.to_tensor_label_motion(label_motion)
            dataset_record.label_motion = label_motion

        return dataset_record
