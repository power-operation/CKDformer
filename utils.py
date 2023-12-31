import os
import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_pck(predictions, ground_truth, thresholds, epoch):
    predictions = predictions.view(predictions.size(0), 17, 2)
    ground_truth = ground_truth.view(ground_truth.size(0), 17, 2)
    num_samples = predictions.size(0)
    num_keypoints = predictions.size(1)

    # Compute the Euclidean distance between each keypoint
    distances = torch.norm(predictions - ground_truth, dim=2)

    pck_dict = []
    # A list is created for storing the PCK value of each keypoint at each threshold
    individual_pck_dict = [[] for _ in range(num_keypoints)]


    # The PCK metric is calculated for each threshold
    for threshold in thresholds:
        correct_keypoints = torch.sum(distances <= threshold, dim=0)
        pck = torch.mean(correct_keypoints.float() / num_samples)
        pck_dict.append(pck)

        if epoch == 199:
            # The PCK value of each keypoint at the current threshold is calculated
            for keypoint_idx in range(num_keypoints):
                individual_pck = torch.mean(correct_keypoints[keypoint_idx].float() / num_samples)
                individual_pck_dict[keypoint_idx].append(individual_pck)

    return pck_dict, individual_pck_dict

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def norm(x):

    n = np.linalg.norm(x)
    return x / n


def student_eval(t_model, s_model, val_loader, args):
    s_model.eval()
    s_high_pressure_loss_record = AverageMeter()
    s_low__pressure_loss_record = AverageMeter()
    s_logits_loss_record = AverageMeter()
    s_acc_record = AverageMeter()

    for img, target in val_loader:
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            t_out, t_high_pressure_encoder_out, t_low_pressure_encoder_out, _ = t_model.forward(
                img, bb_grad=False, output_decoder=False, output_encoder=True)

        s_out, s_high_pressure_encoder_out, s_low_pressure_encoder_out, _ = s_model.forward(
            img, bb_grad=True, output_decoder=False, output_encoder=True)

        logits_loss = F.cross_entropy(s_out, target)

        high_loss = F.kl_div(
            F.log_softmax(s_high_pressure_encoder_out / args.low_T, dim=1),
            F.softmax(t_high_pressure_encoder_out / args.low_T, dim=1),
            reduction='batchmean'
        ) * args.low_T * args.low_T

        low_loss = F.kl_div(
            F.log_softmax(s_low_pressure_encoder_out / args.high_T, dim=1),
            F.softmax(t_low_pressure_encoder_out / args.high_T, dim=1),
            reduction='batchmean'
        ) * args.high_T * args.high_T

        s_high_pressure_loss_record.update(high_loss.item(), img.size(0))
        s_low__pressure_loss_record.update(low_loss.item(), img.size(0))
        s_logits_loss_record.update(logits_loss.item(), img.size(0))
        acc = accuracy(s_out.data, target)[0]
        s_acc_record.update(acc.item(), img.size(0))
    return s_high_pressure_loss_record, s_logits_loss_record, s_low__pressure_loss_record, s_acc_record