from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from dataloader import MixedDataset
from model import MyAwesomeModel


@torch.no_grad()
def infer(args):
    # define dataset and dataloader
    infer_save_dir = Path(args.infer_save_dir)
    infer_dataset = MixedDataset(root_dir=args.dataset_dir,
                    img_paths_file=args.infer_file,
                    img_paths_glob=args.infer_glob,
                    infer=True)
    infer_loader = DataLoader(infer_dataset,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False)

    # define model and load weights
    model = MyAwesomeModel()
    # model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # run inference
    with tqdm(total=len(infer_dataset), desc='Running Inference on images') as pbar:
        for infer_batch in infer_loader:
            index_batch = infer_batch.index
            img_batch = infer_batch.img
            img_prev_batch = infer_batch.img_prev

            output_batch = model(img_batch, img_prev_batch)
            # get motion mask from prediction
            # get the class with the highest probability
            _, motion_masks_batch = torch.max(output_batch, dim=1)

            # save predictions to disk
            for i in range(len(motion_masks_batch)):
                index = index_batch[i]
                motion_mask = output_batch[i]
                img_name = infer_dataset.img_paths[index]
                motion_mask_img = transforms.ToPILImage()(motion_mask)
                motion_mask_img.save(infer_save_dir / img_name)

            pbar.update(len(index_batch))
