# arguments
# --pretrained_model    ["bert", "phobert"]
# --head_model          ["fc", "lstm", "gru", "lstm-attn", "gru-attn", "lstm-cnn", "gru-cnn", "cnn", "transformer"]
# --dataset             ["aivivn", "tiki"]

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

def strToBool(str):
    return str.lower() in ('true', 'yes', 'on', 't', '1')

parser = argparse.ArgumentParser()
parser.register('type', 'bool', strToBool)

parser.add_argument('--pretrained_model', type=str)
parser.add_argument('--head_model', type=str)
parser.add_argument('--dataset', type=str)

parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--nrows', default=None, type=int)

parser.add_argument('--bert_only', type='bool', default=False)
parser.add_argument('--phobert_only', type='bool', default=False)

args = parser.parse_args()

if args.pretrained_model == "bert":
    from BERT.main import train
    train(args.head_model, args.dataset, args.num_epochs, args.batch_size, args.learning_rate, args.nrows, args.bert_only, args.phobert_only)
