"""
computes the table from a Network
@Author: Nathan Greffe
"""

import argparse
from models.wideresnet import *

parser = argparse.ArgumentParser(description='Computing table')
parser.add_argument('--save_file', default='saveto', type=str, help='file in which to save the table')
parser.add_argument('--net', choices=['res'], default='res')
parser.add_argument('--depth', default=40, type=int, help='depth of net')
parser.add_argument('--width', default=2.0, type=float, help='widen_factor of wideresnet')
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', type=str,
                    help='device to use to compute the table')

args = parser.parse_args()
device = torch.device(args.device)
print(device)

if args.net == 'res':
    model = WideResNet(args.depth, args.width, device)
else:
    raise ValueError('pick a valid net')


if __name__ == '__main__':
    model.compute_table(args.save_file)
