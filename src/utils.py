import numpy as np
import torch

# Calculate classification accuravy
def evaluation(nc_logits, labels):

    if len(labels.size()) == 2:
        preds = torch.round(torch.sigmoid(nc_logits))
        tp = len(torch.nonzero(preds * labels, as_tuple=False))
        tn = len(torch.nonzero((1-preds) * (1-labels), as_tuple=False))
        fp = len(torch.nonzero(preds * (1-labels), as_tuple=False))
        fn = len(torch.nonzero((1-preds) * labels, as_tuple=False))
        pre, rec, f1 = 0., 0., 0.
        if tp+fp > 0:
            pre = tp / (tp + fp)
        if tp+fn > 0:
            rec = tp / (tp + fn)
        if pre+rec > 0:
            accuracy = (2 * pre * rec) / (pre + rec)
    else:
        preds = torch.argmax(nc_logits, dim=1)
        correct = torch.sum(preds == labels)
        accuracy = correct.item() / len(labels)

    return accuracy

def SetSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)