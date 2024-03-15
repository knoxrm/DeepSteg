## Metrics
from torch import nn
import torch.distributed as dist
import torch
import numpy as np
import random
import sys
# sys.path.insert(1,'./')
import os
from sklearn.metrics import roc_curve, auc
# from sklearn import metrics
import numpy as np

# def log_args(args, file_path):   
#     with open(file_path, 'a+') as logger:
#         for k,v in args.items():
#             logger.write(f'{k}:{v}\n')
    


def wauc(y_true, y_valid):
    """
    https://www.kaggle.com/anokas/weighted-auc-metric-updated
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        # print(f"idx: {idx} rank: {dist.get_rank()}")
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        if mask.sum()==0:
            continue
        x_padding = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min
        score = auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric
        # print(f"comp met: {competition_metric} rank: {dist.get_rank()}")

    # print("before return")
    return competition_metric / normalization

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0 
        self.count = 0

    def update(self, val, n=1):
        # val = val.clone().detach() if isinstance(val, torch.Tensor) else torch.tensor(val, device="cuda")
        # n = torch.tensor(n, device="cuda") if not isinstance(n, torch.Tensor) else n.clone().detach()
        # val = val.clone().detach() if isinstance(val, torch.Tensor) else torch.tensor(val, device="cuda")
        # n = torch.tensor(n, device="cuda") if not isinstance(n, torch.Tensor) else n.clone().detach()


        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # Synchronize across processes
        # dist.all_reduce(self.sum, op=dist.ReduceOp.SUM)
        # dist.all_reduce(self.count, op=dist.ReduceOp.SUM)

        # Compute average locally
        self.avg = self.sum / self.count


        
class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])
        self.score = torch.tensor(0.0)

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        y_pred = 1 - nn.functional.softmax(y_pred.double(), dim=1).data.cpu().numpy()[:,0]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = torch.tensor(wauc(self.y_true, self.y_pred))
        # sync_score()

    def sync_score(self, device):
        """
        Synchronizes the score across all processes.
        This function should be called after all processes have updated their scores.
        """

        # Summing up scores from all processes
        # sum_score = torch.tensor(self.score).detach().clone()
        sum_score = self.score.clone().detach().to(device)
        dist.all_reduce(sum_score, op=dist.ReduceOp.SUM)
        sum_score /= dist.get_world_size()  # Average the score
        self.score = sum_score.item()

    
    @property
    def avg(self):
        return self.score
