import argparse
import torch
import os
import Models

from torch.utils.data import DataLoader
from ImageDataset import ImageDataset


def main():
    param = param_parser()

    if os.path.exists(param.t):
        raise ValueError('task folder exists')

    images = ImageDataset(param.f, param.w, param.h)

    loader = DataLoader(images, param.b, shuffle=True)

    device = param.d

    if param.dict == '':
        model = Models.get_model(param.m, device, len(images.enumerates))

    else:
        model = Models.build_model(
            torch.load(param.dict, weights_only=False)
        )

    optimizer = torch.optim.SGD(model.parameters(), lr=param.l)

    loss_f = torch.nn.CrossEntropyLoss().to(device)

    os.makedirs(param.t, exist_ok=True)

    for step in range(param.e):

        loss_val = 0.0

        for data, target in loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss = loss_f(output, target)

            loss_val = loss.item()

            loss.backward()

            optimizer.step()

        save_path = '%s/epoch_%d.pth' % (param.t, step)

        save_val = {
            'loss': loss_val,
            'state_dict': model.state_dict(),
            'model': param.m,
            'device': device,
            'out_size': model.output_size,
            'line_input_size': model.line_input_size,
            'w': param.w,
            'h': param.h,
        }

        torch.save(save_val, save_path)

        print('train epoch:%d done loss %f' % (step, loss_val))


def param_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--f', type=str, required=True, help="image folder")

    parser.add_argument('--t', type=str, required=True, help="task folder")

    parser.add_argument('--w', type=int, required=False, default=28, help='resize width')

    parser.add_argument('--h', type=int, required=False, default=28, help='resize height')

    parser.add_argument('--b', type=int, required=False, default=256, help='batch size')

    parser.add_argument('--e', type=int, required=False, default=5, help='epoch')

    parser.add_argument('--d', type=str, required=False, default='cuda:0', help='device')

    parser.add_argument('--m', type=str, required=False, default='LeNet5', help='model')

    parser.add_argument('--l', type=float, required=False, default=0.01, help='learning rate')

    parser.add_argument('--dict', type=str, required=False, default='', help='state dict')

    return parser.parse_args()


if __name__ == "__main__":
    main()
