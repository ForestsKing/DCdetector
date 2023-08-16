import argparse

import torch

from exp.exp import Exp
from utils.fix_seed import fix_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./dataset/')
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--channel', type=int, default=33)

    parser.add_argument('--window_size', type=int, default=105)
    parser.add_argument('--patch_size', type=list, default=[5, 7])
    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--head_num', type=int, default=1)
    parser.add_argument('--encoder_layer', type=int, default=3)
    parser.add_argument('--anomaly_ratio', type=float, default=5)

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--fix_seed', type=int, default=42)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--devices', type=int, default=0)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('\n=====================Args========================')
    print(args)
    print('=================================================\n')

    fix_seed(args.fix_seed)

    setting = 'ws{0}_ps{1}_md{2}_hn{3}_el{4}_ar{5:.2f}'.format(
        args.window_size, args.patch_size, args.model_dim, args.head_num, args.encoder_layer, args.anomaly_ratio
    )

    print('\n>>>>>>>>  initing : {}  <<<<<<<<\n'.format(setting))
    exp = Exp(args, setting)

    print('\n>>>>>>>>  training : {}  <<<<<<<<\n'.format(setting))
    exp.train()

    print('\n>>>>>>>>  testing : {}  <<<<<<<<\n'.format(setting))
    exp.test()

    print('Done!')
