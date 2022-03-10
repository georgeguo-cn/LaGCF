#!/usr/bin/python -u
"""
    @time:Created on April 2, 2021
    @author: Zhiqiang Guo (zhiqiangguo@hust.edu.cn)
    Pytorch Implementation of LaGCF in 'Zhiqiang Guo et al. Joint Locality Preservation and Adaptive Combination for Graph Collaborative Filtering'
"""

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="a new idea of using residual GCN for RSs")
    parser.add_argument('--seed', type=int, default=2021, help='Random seed.')
    parser.add_argument('--gpu_id', type=int, default=0, help="if -1 use cpu")
    parser.add_argument('--dataset', type=str, default='Ciao', help="datasets: [Amazon-books, Amazon-video, Brightkite, Ciao, Gowalla, ml-1m, Yelp]")
    parser.add_argument('--topks', nargs='?', default="[10, 20]", help="@k test list")
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--bpr_batch', type=int, default=2048, help="the batch size for bpr loss training procedure")

    parser.add_argument('--model', type=str, default='lagcf', help='rec-model, support [mf, mlp, ncf, lgn, ngcf, lagcf]')
    parser.add_argument('--load_model', type=bool, default=False, help='whether we use saved model or not')
    parser.add_argument('--save_emb', type=bool, default=False, help='whether we save the emb of user and item or not')
    parser.add_argument('--initilizer', type=str, default='normal', help='the initilizer for embedding, support [xavier, normal, pretrain]')
    parser.add_argument('--emb', type=int, default=64, help="the embedding size")
    parser.add_argument('--layer', type=int, default=4, help="the layer num")
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--decay', type=float, default=0.0001, help="the weight decay for L2 normalizaton")
    parser.add_argument('--norm', type=str, default='l2', help="the normalizaton for weight, support [l1, l2]")
    parser.add_argument('--LP', type=str, default='FA', help='local preserving type, support [ZA,FA]')
    parser.add_argument('--IM', type=str, default='PM', help='identity mapping type, support [PN,PM]')
    parser.add_argument('--AGG', type=str, default='LS', help='aggregation type, support [L,S,LS]')

    parser.add_argument('--tensorboard', type=bool, default=False, help="enable tensorboard")
    parser.add_argument('--early_stop', type=int, default=50, help='early_stop')
    parser.add_argument('--alpha', type=float, default=0.001, help='alpha')
    return parser.parse_args()
