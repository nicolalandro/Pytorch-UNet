import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from daedalus_dataset import DaedalusDataset


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--output', '-o', help='folders of output images')

    return parser.parse_args()


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    output_folder = args.output

    net = UNet(n_channels=2, n_classes=2)
    net.eval()

    device = 'cpu'
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    dataset = DaedalusDataset(
        '/home/supreme/datasets-nas/INAF/daedalus/dataset-pairs/test',
    )

    to_pil = transforms.ToPILImage()

    for i, data in enumerate(dataset):
        id = data['idx']
        img = data['image'].unsqueeze(dim=0)
        with torch.no_grad():
            output = net(img)
            probs = F.softmax(output, dim=1)[0]
        mask_generated = probs.argmax(dim=0).unsqueeze(dim=0)*1.0
        result = to_pil(mask_generated)
        
        file_name = dataset.label_name_list[id.item()]
        out_path = os.path.join(output_folder, file_name)
        result.save(out_path)
        print(f'Mask saved to {out_path}')
        if i > 5:
            break