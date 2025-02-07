import torch
from torch import nn


class LeNet5(nn.Module):

    def __init__(self, device, output_size):
        super(LeNet5, self).__init__()

        self.linear = None
        self.device = device
        self.output_size = output_size
        self.line_input_size = 0

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(6),

            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
        )
        self.to(device)

    def build_linear(self, size):
        if self.linear is not None:
            return

        self.line_input_size = size
        self.linear = nn.Sequential(
            nn.Linear(size, 100),
            torch.nn.LeakyReLU(inplace=True),
            nn.Linear(100, 50),
            torch.nn.LeakyReLU(inplace=True),
            nn.Linear(50, self.output_size),
            torch.nn.LeakyReLU(inplace=True),
        )

        self.to(self.device)

    def forward(self, x):
        size = x.size(dim=0)

        res = self.conv(x).view(size, -1)

        _, size = res.shape

        self.build_linear(size)

        return self.linear(res)


def get_model(name, device, out_size):
    match name:
        case 'LeNet5':
            return LeNet5(device, out_size)

    raise ValueError('err model:' + name)


def build_model(model_param):
    model = get_model(model_param['model'], model_param['device'], model_param['out_size'])
    
    model.build_linear(model_param['line_input_size'])
    model.load_state_dict(model_param['state_dict'])

    return model
