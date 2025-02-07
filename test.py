import argparse
import Models
import torch

from ImageDataset import ImageDataset
from torch.utils.data import DataLoader


def main():
    param = param_parser()

    model_param = torch.load(param.dict, weights_only=False)

    model = Models.build_model(model_param)
    model.eval()

    err = 0

    with torch.no_grad():
        images = ImageDataset(param.f, model_param['w'], model_param['h'])
        loader = DataLoader(images, 1000, shuffle=True)

        for data, targets in loader:
            data, targets = data.to(model_param['device']), targets.to(model_param['device'])
            predicts = model(data)

            _, predicts = torch.max(predicts, dim=1)

            for predict, target in zip(predicts, targets):
                if predict != target:
                    err += 1

        print('test done count:%d err:%d' % (len(images), err))


def param_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--f', type=str, required=True, help="image folder")

    parser.add_argument('--dict', type=str, required=True, default='', help='state dict')

    return parser.parse_args()


if __name__ == "__main__":
    main()
