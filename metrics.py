import numpy as np


class IoU:
    """Computes the intersection over union (IoU) per class and corresponding mean (mIoU).
    The predictions are first accumulated in a confusion matrix and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    :param num_classes (int): number of classes in the classification problem
    :param dataset (string): woodscape_raw
    :param ignore_index (int or iterable, optional): Index of the classes to ignore when computing the IoU.
    """

    def __init__(self, num_classes, ignore_index=None, weights=None):
        super().__init__()

        self.conf_matrix = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.num_classes = num_classes
        self.weights = weights if weights is not None else np.ones(num_classes) / num_classes
        self.reset()

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_matrix.fill(0)

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric."""

        predicted = predicted.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        # hack for bin counting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes ** 2)
        assert bincount_2d.size == self.num_classes ** 2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        self.conf_matrix += conf

    def value(self):
        """Computes the IoU and mean IoU.
        The mean computation ignores NaN elements of the IoU array.
        Returns: Tuple: (class_iou, mIoU). The first output is the per class IoU, for K classes it's numpy.ndarray with
        K elements. The second output, is the mean IoU.
        """
        if self.ignore_index is not None:
            for index in self.ignore_index:
                self.conf_matrix[:, self.ignore_index] = 0
                self.conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(self.conf_matrix)
        false_positive = np.sum(self.conf_matrix, 0) - true_positive
        false_negative = np.sum(self.conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        class_names = ['static', 'motion']
        class_iou = dict(zip(class_names, iou))
        weighted_iou = np.average(iou, weights=self.weights)

        return class_iou, np.mean(iou), weighted_iou
