import numpy as np

class mIoU():
    """
    Accumulate results to caluculate final mIoU
    """
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns score evaluation result.
            - overall accuracy
            - mean IU
        """

        hist = self.confusion_matrix

        iou_0 = hist[0,0] /(hist[0,0] + hist[0,1] + hist[1,0])
        iou_1 = hist[1,1] /(hist[1,1] + hist[0,1] + hist[1,0])
        mean_iu = (iou_1 + iou_0)/2

        cls_iu = {0: iou_0, 1: iou_1}
        return {
            "Total samples": self.total_samples,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0