import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GL-Disen Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=2,
            help='Number of graph convolution layers before disentangling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=128,
            help='')

    return parser.parse_args()

