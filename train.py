import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from data import get_dataloader
from models import model_dict
import os
from utils import AverageMeter, calculate_pck
import numpy as np
from datetime import datetime
import time
import sinkhorn as spc

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_names', type=str, nargs='+', default=['conti', 'conti'])
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.1, help='weight for confidence loss')

parser.add_argument('--root', type=str, default='dataset')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--epoch', type=int, default=200)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--pr', type=bool, default=False)
parser.add_argument('--sa', type=bool, default=False)
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu-ids', type=str, default='0', help='Comma-separated list of GPU IDs')
parser.add_argument('--print_freq', type=int, default=100)

args = parser.parse_args()
args.num_branch = len(args.model_names)

threshold = [10, 20, 30, 40, 50]

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
device_ids = [0]

exp_name = '_'.join(args.model_names)
exp_path = './experiments/{}/{}'.format(exp_name, datetime.now().strftime('%Y-%m-%d-%H-%M'))
os.makedirs(exp_path, exist_ok=True)
print(exp_path)

def train_one_epoch(models, optimizers, train_loader, epoch):
    pck_recorder_list = [list() for _ in range(5)]
    pckess = []

    loss_recorder_list = []
    for model in models:
        model.train()
        for i in range(0,5):
            pck_recorder_list[i].append(AverageMeter())
        loss_recorder_list.append(AverageMeter())

    for i, (imgs, label) in enumerate(train_loader):
        outputs = torch.zeros(size=(len(models), imgs.size(0), 34), dtype=torch.float).cuda()
        out_list = []

        # forward
        for model_idx, model in enumerate(models):

            if torch.cuda.is_available():
                imgs = imgs.float().cuda()
                label = label.cuda()

            if args.model_names[0] == 'conti':
                out = model.forward(imgs[:, model_idx, ...])

            else:
                out = model.forward(imgs[:, model_idx, ...])
            
            outputs[model_idx, ...] = out
            out_list.append(out)

        # backward
        stable_out = outputs.mean(dim=0)
        stable_out = stable_out.detach()

        for model_idx, model in enumerate(models):
            ce_loss = F.l1_loss(out_list[model_idx].to(torch.float16), label)
            # Sinkhorn parameters
            epsilon = 0.01
            niter = 100

            div_loss = spc.sinkhorn_loss(stable_out, out_list[model_idx], epsilon, stable_out.shape[0], niter)
                       
            if len(models) == 1:
                loss = ce_loss
            else:
                loss = (1 - args.alpha) * ce_loss + (args.alpha) * div_loss

            optimizers[model_idx].zero_grad()
            if model_idx < len(models) - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            optimizers[model_idx].step()

            loss_recorder_list[model_idx].update(loss.item(), n=imgs.size(0))
            
            pcks, _ = calculate_pck(out_list[model_idx], label, threshold, epoch)
            pck = [round(float(x), 4) for x in pcks]
            for idx, _ in enumerate(threshold):
                pck_value = pck[idx]
                pck_recorder_list[idx][model_idx].update(pck_value, n=imgs.size(0))

    losses = [recorder.avg for recorder in loss_recorder_list]
    for i in range(0, 5):
        pckes = [recorder.avg for recorder in pck_recorder_list[i]]
        pckess.append(pckes)
    pcky = [[round(x, 4) for x in sublist] for sublist in pckess]

    return losses, pcky

def evaluation(models, val_loader, epoch):
    pck_recorder_list = [list() for _ in range(5)]
    pckess = []
    loss_recorder_list = []
    pck_batch_results_model1 = []
    pck_batch_results_model2 = []
    for model in models:
        model.eval()
        for i in range(0,5):
            pck_recorder_list[i].append(AverageMeter())
        loss_recorder_list.append(AverageMeter())

    with torch.no_grad():
        for img, label in val_loader:
            if torch.cuda.is_available():
                img = img.float().cuda()
                label = label.cuda()

            for model_idx, model in enumerate(models):
                if args.model_names[0] == 'conti':
                    out = model(img[:, model_idx, ...])
                else:
                    out = model(img[:, model_idx, ...])

                loss = F.l1_loss(out, label)
                loss_recorder_list[model_idx].update(loss.item(), img.size(0))
                pcks, key_pcks = calculate_pck(out, label, threshold, epoch)
                if epoch == args.epoch - 1:
                    if model_idx == 0:
                        pck_batch_results_model1.append(key_pcks)
                    elif model_idx == 1:
                        pck_batch_results_model2.append(key_pcks)

                pck = [round(float(x), 4) for x in pcks]
                for idx, _ in enumerate(threshold):
                    pck_value = pck[idx]
                    pck_recorder_list[idx][model_idx].update(pck_value, img.size(0))

    losses = [recorder.avg for recorder in loss_recorder_list]
    for i in range(0, 5):
        pckes = [recorder.avg for recorder in pck_recorder_list[i]]
        pckess.append(pckes)
    pcky = [[round(x, 4) for x in sublist] for sublist in pckess]

    return losses, pcky, pck_batch_results_model1, pck_batch_results_model2


def train(model_list, optimizer_list, train_loader, scheduler_list):
    best_pck = [[0,0],[0,0],[0,0],[0,0],[0,0]]
    best_loss = [100 for _ in range(args.num_branch)]

    for epoch in range(args.epoch):
        train_start_time = time.time()
        train_losses, train_pck = train_one_epoch(model_list, optimizer_list, train_loader, epoch)
        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        eval_start_time = time.time()
        val_losses, val_pack, pck_1, pck_2 = evaluation(model_list, val_loader, epoch)
        eval_end_time = time.time()
        eval_time = eval_end_time - eval_start_time

        for i in range(len(best_loss)):
            for j in range(len(best_pck)):
                if val_pack[j][i] > best_pck[j][i]:
                    best_pck[j][i] = val_pack[j][i]

        for i in range(len(best_loss)):
            if val_losses[i] < best_loss[i]:
                best_loss[i] = val_losses[i]
                state_dict = dict(epoch=epoch + 1, model=model_list[i].state_dict(),
                                  acc=val_losses[i])
                set_lst = set(args.model_names)

                if len(set_lst) != len(args.model_names) and args.model_names[0] != 'conti':
                    name = os.path.join(exp_path, args.model_names[i]+str(i), 'ckpt', 'best.pth')
                    os.makedirs(os.path.dirname(name), exist_ok=True)
                    torch.save(state_dict, name)
                else:
                    name = os.path.join(exp_path, args.model_names[i], 'ckpt', 'best.pth')
                    os.makedirs(os.path.dirname(name), exist_ok=True)
                    torch.save(state_dict, name)

            scheduler_list[i].step()

        if (epoch + 1) % args.print_freq == 0:
            for j in range(len(best_loss)):
                print("epoch:{} model:{}{} train_loss:{:.2f} val_loss{:.2f} train_pck:{},{},{},{},{} val_pck:{},{},{},{},{}".format(
                    epoch+1, args.model_names[j], j, train_losses[j], val_losses[j], train_pck[0][j], train_pck[1][j], train_pck[2][j], train_pck[3][j], train_pck[4][j],
                    best_pck[0][j], best_pck[1][j], best_pck[2][j], best_pck[3][j], best_pck[4][j]
                    ))

        if len(args.model_names) == 1:
            print("epoch {} training_time: {:.2f}s test_time: {:.2f}s best_loss: {:.2f}".format(epoch+1, train_time, eval_time, best_loss[0]))
        else:
            print("epoch {} training_time: {:.2f}s test_time: {:.2f}s best_loss1: {:.2f} best_loss2: {:.2f}".format(epoch+1, train_time, eval_time, best_loss[0], best_loss[1]))
        
        if epoch == args.epoch - 1:
            num_keypoints = 17
            num_thresholds = 5
            avg_pck_values_model1 = [[] for _ in range(num_keypoints)]
            avg_pck_values_model2 = [[] for _ in range(num_keypoints)]

            # Iterate over the list of PCK values for each keypoint
            for keypoint_idx in range(num_keypoints):
                # The PCK value at each threshold is traversed
                for threshold_idx in range(num_thresholds):
                    # The PCK values at a specific threshold for a specific keypoint in all batches are extracted and the average value is calculated
                    pck_values_model1 = [pck_result[keypoint_idx][threshold_idx] for pck_result in pck_1]
                    avg_pck_model1 = sum(pck_values_model1) / len(pck_values_model1)
                    avg_pck_values_model1[keypoint_idx].append(avg_pck_model1)

            # Iterate over the list of PCK values for each keypoint
            for keypoint_idx in range(num_keypoints):
                for threshold_idx in range(num_thresholds):
                    pck_values_model2 = [pck_result[keypoint_idx][threshold_idx] for pck_result in pck_2]
                    avg_pck_model2 = sum(pck_values_model2) / len(pck_values_model2)
                    avg_pck_values_model2[keypoint_idx].append(avg_pck_model2)

            # Calculate the average PCK value for each model
            num_keypoints = 17
            num_thresholds = 5
            avg_pck_model1 = [[] for _ in range(num_thresholds)]
            avg_pck_model2 = [[] for _ in range(num_thresholds)]

            # Iterate through the PCK values at each threshold value
            for threshold_idx in range(num_thresholds):
                # Traverse each key point
                for keypoint_idx in range(num_keypoints):
                    # Calculate the average PCK value of model at this keypoint, at this threshold
                    pck_values_model1 = [pck_result[keypoint_idx][threshold_idx] for pck_result in pck_1]
                    avg_pck_model1[threshold_idx].append(sum(pck_values_model1) / len(pck_values_model1))
                    
                    pck_values_model2 = [pck_result[keypoint_idx][threshold_idx] for pck_result in pck_2]
                    avg_pck_model2[threshold_idx].append(sum(pck_values_model2) / len(pck_values_model2))

            # The average of the two models is taken as the final output
            final_avg_pck_values = [[] for _ in range(num_thresholds)]

            for threshold_idx in range(num_thresholds):
                for keypoint_idx in range(num_keypoints):
                    avg_pck = (avg_pck_model1[threshold_idx][keypoint_idx] + avg_pck_model2[threshold_idx][keypoint_idx]) / 2
                    final_avg_pck_values[threshold_idx].append(avg_pck)
            
            final_avg_pck_values = [[round(tensor_item.item(), 4) for tensor_item in tensor_list] for tensor_list in final_avg_pck_values]

            print(final_avg_pck_values)

    for k in range(len(best_loss)):
        print("model:{} best loss:{:.2f}".format(args.model_names[k], best_loss[k]))

if __name__ == '__main__':
    train_loader, val_loader = get_dataloader(args)
    model_list = []
    optimizer_list = []
    scheduler_list = []
    for ind, name in enumerate(args.model_names):
        lr = 0.0001
        if name in ['conti']:
            model = model_dict[name](num_classes=34, ind=ind)
            print("use contimulti!")
        else:
            model = model_dict[name](num_classes=34)
        if torch.cuda.is_available(): 
            model = nn.DataParallel(model, device_ids=device_ids)
            model = model.cuda()
            print("------Several GPU used!------")

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)
        criterion = nn.L1Loss()
        criterion = criterion.cuda()
        model_list.append(model)
        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)

    train(model_list, optimizer_list, train_loader, scheduler_list)