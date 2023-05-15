import os, sys
sys.path.append(os.path.abspath('.'))

import importlib
from glob import glob
import argparse

evaluation = importlib.import_module('evaluation')
test = getattr(evaluation, 'test')

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--tag', type=str)
parser.add_argument('--visualize', '-v', action='store_true', default=False)

args = parser.parse_args()
checkpoint = args.checkpoint
tag = args.tag

ckpt_dir = os.path.join('log', checkpoint)
eval_glob = os.path.join(ckpt_dir, tag)
eval_filenames = sorted(glob(eval_glob))

history = []
for eval_filename in eval_filenames:
    tag = eval_filename.replace(ckpt_dir, '')[1:]
    acc, fr = test(checkpoint=checkpoint, tag=tag, device=args.device, visualize=args.visualize)
    history.append((tag, acc, fr))
print()

for tag, acc, fr in history:
    print(f'{tag}\t{100.0 - acc:.3f}\t{fr:.3f}')
