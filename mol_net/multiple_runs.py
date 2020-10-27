import os, sys
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Multiple runs for node classification')
parser.add_argument('--gpu_id',type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--dataset', type=str, default = 'cora',
                    help='the name of dataset. For now, only classification.')
parser.add_argument('--filename', type=str, default = '',
                    help='the name of output model')
parser.add_argument('--gnn_type', type=str, default="gcn",
                    help="the Graph Neural Network to use")
parser.add_argument('--result_file', type=str, default='',
                    help='the file to store results')
parser.add_argument('--num_run', type=int, default=100,
                    help='the number of independent runs (default: 100)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0,
                    help='weight decay (default: 0)')
# parameters for representation disentanglement
parser.add_argument('--use_embed_disen', action='store_true', default=False,
                    help='whether to use feature disentangle')
parser.add_argument('--lambda_recon', type=float, default=1.0,
                    help='the weight of embedding reconstruction loss')
parser.add_argument('--lambda_MI_min', type=float, default=0.1,
                    help='the weight of mutual information minimization')
# parameters for input disentanglement
parser.add_argument('--use_input_disen', action='store_true', default=False,
                    help='whether to use input disentangle')
parser.add_argument('--emb_dim', type=int, default=150,
                    help='embedding dimensions (default: 150)')

args = parser.parse_args()

if os.path.exists(args.result_file):
    os.remove(args.result_file)

# Independent runs under the same configuration
for run_id in range(args.num_run):
    # Run for Embed-SAD
    if args.use_embed_disen:
        print(args)
        template = 'python3.6 train_embed_SAD_node.py --device {} --dataset {} --filename {} --result_file {} ' \
                   ' --gnn_type {} --lambda_MI_min {} --lambda_recon {}'
        cmd = template.format(args.gpu_id, args.dataset, args.filename, args.result_file, args.gnn_type,
                              args.lambda_MI_min, args.lambda_recon)
        print(cmd)
        os.system(cmd)
    # Run for Input-SAD
    elif args.use_input_disen:
        print(args)
        template = 'python3.6 train_input_SAD_node.py --device {} --dataset {} --filename {} --gnn_type {} ' \
                   ' --result_file {} --emb_dim {}'
        cmd = template.format(args.gpu_id, args.dataset, args.filename, args.gnn_type, args.result_file, args.emb_dim)
        print(cmd)
        os.system(cmd)
    # Run for baseline
    else:
        print(args)
        template = 'python3.6 train_baseline_node.py --device {} --dataset {} --filename {} --gnn_type {} ' \
                   ' --result_file {}'
        cmd = template.format(args.gpu_id, args.dataset, args.filename, args.gnn_type, args.result_file)
        print(cmd)
        os.system(cmd)

# Compute performance
acc_list = []
result_file = open(args.result_file, 'r')
line = result_file.readline()
while line:
    acc = float(line.strip())
    acc_list.append(acc)
    line = result_file.readline()

result_file.close()

acc_np_list = np.array(acc_list)
print ('Mean: {}\tStd: {}'.format(acc_np_list.mean(), acc_np_list.std()))

end_cmd = 'watch nvidia-smi'
os.system(end_cmd)
