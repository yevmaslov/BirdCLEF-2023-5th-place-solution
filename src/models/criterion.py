import pandas as pd
import torch.nn as nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=6, weights=None):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored
        self.weights = [1/num_scored for _ in range(num_scored)] if weights is None else weights

    def forward(self, yhat, y):
        score = 0
        for i, w in enumerate(self.weights):
            score += self.rmse(yhat[:, :, i], y[:, :, i]) * w
        return score


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor.argmax(dim=1),
            weight=self.weight,
            reduction=self.reduction,
        )


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class WeightedDenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)

        if weights is not None:
            loss = loss * weights
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.mean()

        return loss

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1 - label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive


def cosine_distance(x1, x2):
    return 1 - torch.nn.functional.cosine_similarity(x1, x2)


class MixupCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        pass
    
    def forward(self, input, labels):
        assert input.size() == target.size()
        assert isinstance(input, Variable) and isinstance(target, Variable)
        input = torch.log(torch.nn.functional.softmax(input, dim=1).clamp(1e-5, 1))
        # input = input - torch.log(torch.sum(torch.exp(input), dim=1)).view(-1, 1)
        loss = - torch.sum(input * labels)
        return loss / input.size()[0] if size_average else loss
    
    
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss



def get_criterion(config):
    if config.criterion.type == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif config.criterion.type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError
