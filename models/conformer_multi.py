import torch
import torch.nn as nn
import torchaudio

class conformer(nn.Module):
    def __init__(self):
        super(conformer, self).__init__()
        self.conformer = torchaudio.models.Conformer(
            input_dim=12*51, 
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
            depthwise_conv_kernel_size=15,
        )

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), -1)
        lengths = torch.ones(x.size(0)).cuda()
        shared_features = self.conformer(x, lengths)[0].squeeze(dim=1)
        return shared_features

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, input_dim, ind, num_predict_list):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        self.num_tasks = len(num_predict_list)
        self.task_branches = nn.ModuleList()
        self.ind = ind
        self.task_branches.append(nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 34)
        ))
        self.task_branches.append(nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 34)
        ))


    def forward(self, x):
        features = self.backbone(x)
        if self.ind == 0:
            outputs = self.task_branches[0](features)
        if self.ind == 1:
            outputs = self.task_branches[1](features)

        return outputs


def conformermulti(num_classes, ind):
    backbone = conformer()
    num_predict_list = [num_classes, num_classes]  # the num of num_classes should be ajust when model num change
    input_dim = 12*51
    model = MultiTaskModel(backbone, input_dim, ind, num_predict_list)
    return model

if __name__ == '__main__':
    x = torch.randn(2, 1, 32, 32) 
    model = conformermulti(num_classes=34)
    output = model(x)
    print(output.shape)
