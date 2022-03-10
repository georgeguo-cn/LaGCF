#!/usr/bin/python -u
"""
    @time:Created on April 2, 2021
    @author: Zhiqiang Guo (zhiqiangguo@hust.edu.cn)
    Pytorch Implementation of LaGCF in 'Zhiqiang Guo et al. Joint Locality Preservation and Adaptive Combination for Graph Collaborative Filtering'
"""
import os
import sys
import torch
from time import time
import random
import warnings
import numpy as np
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter

import utils
import parse
import model
import dataloader

# ignore warnings
warnings.filterwarnings('ignore')
# help plt.imshow()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

args = parse.parse_args()
if args.model in ['mlp', 'ncf']:
    args.layer = 3
print(args)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

str_dev = "cuda:" + str(args.gpu_id)
device = torch.device(str_dev if (torch.cuda.is_available() and args.gpu_id >= 0) else "cpu")
print('Device:', device)

root_path = "./"
data_path = root_path + 'data/'
model_path = root_path + 'checkpoints/'
log_path = root_path + 'logs/'
run_path = root_path + 'runs/'
emb_path = root_path + 'embs/'

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
# logfile
_log = open(log_path + args.dataset + '_' + args.model + '_'
            + str(args.emb) + '_' + str(args.layer) + '_' + str(args.alpha) + '_'
            + args.LP + '-' + args.IM + '-' + args.AGG + '_' + datetime.now().strftime("%m%d%H%M%S") + '.txt', 'w')
_log.write(str(args) + '\n')

all_dataset = ['ABooks', 'Ciao', 'Gowalla', 'ml-1m', 'Yelp']
all_models = ['mf', 'mlp', 'ncf', 'ngcf', 'lgn', 'lagcf']
dataname = args.dataset
model_name = args.model
if dataname not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataname} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

topks = eval(args.topks)
MODELS = {'mf': model.MF, 'mlp': model.MLP, 'ncf': model.NeuMF, 'ngcf': model.NGCF,
          'lgn': model.LightGCN, 'lagcf': model.LaGCF, 'sgcf': model.SimGCF}

if __name__ == '__main__':
    dataset = dataloader.Loader(args, dataname, device)
    Rmodel = MODELS[model_name](args, dataset)
    Rmodel = Rmodel.to(device)
    optimizer = optim.Adam(Rmodel.parameters(), lr=args.lr)

    # save and load model
    if args.load_model:
        model_file = model_path + args.dataset + '_' + args.model + '_' \
                     + str(args.emb) + '_' + str(args.layer) + '_' \
                     + str(args.alpha) + '.pth.tar'
        print('load and save to', model_file)
        if os.path.exists(model_file):
            Rmodel.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
            print('loaded model weights from', model_file)
        else:
            print(f"{model_file} not exists, start from beginning")

    if args.save_emb:
        if not os.path.exists(emb_path):
            os.makedirs(emb_path)
        emb_file = emb_path + args.dataset + '_' + + args.model + '_' + str(args.layer) \
                   + '_' + args.LP + '-' + args.IM + '-' + args.AGG + '_'

    # init tensorboard
    if args.tensorboard:
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        w: SummaryWriter = SummaryWriter(run_path + args.dataset + '_' + args.model + '_'
                                         + str(args.emb) + '_' + str(args.layer) + '_' + str(args.alpha) + '_'
                                         + datetime.now().strftime("%m%d%H%M%S"))
    else:
        w = None
        print("not enable tensorboard")

    max_value = 0.0
    max_epoch = 0
    step = 0
    best_results = {'precision': np.zeros(len(topks)), 'recall': np.zeros(len(topks)), 'ndcg': np.zeros(len(topks))}
    flag = False
    losses = []
    for epoch in range(args.epochs):
        if epoch % 10 == 0:
            Rmodel.simGraph()
        if epoch % 10 == 0:
            test_time = time()
            results, emb_users, emb_items = utils.Test(args, dataset, Rmodel, epoch, w, device)
            str_log = 'Epoch[' + str(epoch+1) + '] ' + str(results).replace('array(', '').replace(')', '')
            _log.write(str_log + '\n')
            _log.flush()
            if max_value <= results['recall'][-1]:
                max_value = results['recall'][-1]
                max_epoch = epoch + 1
                step = 0
                if args.save_emb:
                    np.save(emb_file +'embeddings_user.npy', emb_users)
                    np.save(emb_file + 'embeddings_item.npy', emb_items)
            for key, value in best_results.items():
                best_results[key] = np.maximum(value, results[key])
            # torch.save(Rmodel.state_dict(), model_file)
            print(f'[TEST {round(time()-test_time, 2)}s], {str(results).replace("array(","").replace(")", "")}')
        start = time()
        Rmodel.train()
        S = utils.UniformSample_original(dataset)
        users = torch.Tensor(S[:, 0]).long().to(device)
        posItems = torch.Tensor(S[:, 1]).long().to(device)
        negItems = torch.Tensor(S[:, 2]).long().to(device)
        users, posItems, negItems = utils.shuffle(users, posItems, negItems)
        total_batch = len(users) // args.bpr_batch + 1
        aver_loss = 0.
        for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(users, posItems, negItems, batch_size=args.bpr_batch)):
            optimizer.zero_grad()
            users_emb, pos_emb, neg_emb = Rmodel(batch_users, batch_pos, batch_neg)
            bpr_loss, reg_loss = Rmodel.bpr_loss(batch_users, batch_pos, batch_neg, users_emb, pos_emb, neg_emb)
            loss = bpr_loss + reg_loss * args.decay
            loss.backward()
            optimizer.step()
            aver_loss += loss
            if args.tensorboard:
                w.add_scalar(f'BPRLoss/BPR', aver_loss/(batch_i+1), epoch * int(len(users) / args.bpr_batch) + batch_i)
        aver_loss = aver_loss / total_batch
        losses.append(aver_loss.cpu().item())
        print(f'EPOCH {epoch + 1}/{args.epochs} [{round(time()-start, 2)}s], loss: {aver_loss:.5f}')

        if step > args.early_stop and epoch > 500:
            break
        step += 1

    print('BestResult:', str(best_results).replace('array(', '').replace(')', ''), 'max epoch:', max_epoch)
    _log.write('BestResults: ' + str(best_results).replace('array(', '').replace(')', '') + '\n')
    _log.write('loss:' + str(losses))
    _log.flush()
    _log.close()