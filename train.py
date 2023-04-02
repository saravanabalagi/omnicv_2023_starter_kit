from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from dataloader import MixedDataset
from model import MyAwesomeModel
from metrics import IoU
from utils import print_tensor_info


def train(args):
    # define dataset and dataloader
    dataset_train = MixedDataset(root_dir=args.dataset_dir, img_paths_file=args.train_file)
    dataloader_train = DataLoader(dataset_train,
                        num_workers=args.num_workers,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True)
    dataset_val = MixedDataset(root_dir=args.dataset_dir, img_paths_file=args.val_file)
    dataloader_val = DataLoader(dataset_val,
                        num_workers=args.num_workers,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True)

    # define model and load weights
    model = MyAwesomeModel()
    criterion = torch.nn.CrossEntropyLoss()
    metric_motion = IoU(num_classes=2, weights=[1/20, 19/20])
    # optimizer = None

    for epoch in tqdm(range(args.num_epochs), desc='Training'):
        train_batch(model, dataloader_train, criterion, metric_motion, epoch)
        val_batch(model, dataloader_val, metric_motion, epoch)
        # torch.save(model.state_dict(), args.model_path)


def train_batch(model, dataloader, criterion, metric_motion, epoch):
    for batch_number, train_batch in enumerate(dataloader):
        model.train()
        index_batch = train_batch.index
        img_batch = train_batch.img
        img_prev_batch = train_batch.img_prev
        label_motion_batch = train_batch.label_motion
        label_motion_mask_batch = (label_motion_batch >= 1).long()

        output_batch = model(img_batch, img_prev_batch)
        # get motion mask from prediction
        # get the class with the highest probability
        _, pred_motion_masks_batch = torch.max(output_batch, dim=1)

        # for debugging TODO: remove
        if batch_number == 0:
            print(f'Loaded images {[dataloader.dataset.img_paths[i] for i in index_batch]}')
            print_tensor_info(img_batch, 'img batch')
            print_tensor_info(img_prev_batch, 'img_prev batch')
            print_tensor_info(label_motion_batch, 'label_motion batch')
            print_tensor_info(output_batch, 'output batch')
            print_tensor_info(pred_motion_masks_batch, 'pred_motion_mask batch')
            print_tensor_info(label_motion_mask_batch, 'label_motion_mask batch')

        metric_motion.add(pred_motion_masks_batch, label_motion_mask_batch)
        loss = criterion(output_batch, label_motion_mask_batch)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    class_iou, mean_iou, weighted_iou = metric_motion.value()
    class_iou_str = ', '.join([f'{k}: {v:.4f}' for k, v in class_iou.items()])
    print(f'Epoch {epoch} train IoU: {weighted_iou:.4f} ({class_iou_str}), mean {mean_iou:.4f}')
    metric_motion.reset()


@torch.no_grad()
def val_batch(model, dataloader, metric_motion, epoch):
    for val_batch in dataloader:
        model.eval()
        index_batch = val_batch.index
        img_batch = val_batch.img
        img_prev_batch = val_batch.img_prev
        label_motion_batch = val_batch.label_motion
        label_motion_mask_batch = (label_motion_batch >= 1).long()

        output_batch = model(img_batch, img_prev_batch)
        # get motion mask from prediction
        # get the class with the highest probability
        _, pred_motion_mask_batch = torch.max(output_batch, dim=1)

        metric_motion.add(pred_motion_mask_batch, label_motion_mask_batch)

    class_iou, mean_iou, weighted_iou = metric_motion.value()
    class_iou_str = ', '.join([f'{k}: {v:.4f}' for k, v in class_iou.items()])
    print(f'Epoch {epoch} val IoU: {weighted_iou:.4f} ({class_iou_str}), mean {mean_iou:.4f}')
    metric_motion.reset()
