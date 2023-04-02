import torch
import torch.nn as nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # define your model here

    def forward(self, img, img_prev):
        # define forward pass here
        # predict probabilities for each class
        # let's predict for 2 classes: for static and dynamic
        pred_shape = (img.shape[0], 2, img.shape[2], img.shape[3])
        pred = torch.rand(pred_shape)
        return pred

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
