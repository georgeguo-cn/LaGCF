#!/usr/bin/python -u
"""
    @time:Created on April 2, 2021
    @author: Zhiqiang Guo (zhiqiangguo@hust.edu.cn)
    Pytorch Implementation of LaGCF in 'Zhiqiang Guo et al. Joint Locality Preservation and Adaptive Combination for Graph Collaborative Filtering'
    Define models here
"""
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import parse
args = parse.parse_args()
str_dev = "cuda:" + str(args.gpu_id)
device = torch.device(str_dev if (torch.cuda.is_available() and args.gpu_id >= 0) else "cpu")

class MF(nn.Module):
    def __init__(self, config, dataset):
        super(MF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config.emb
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_item.weight, std=0.01)
        print("using Normal distribution N(0,0.01) initialization for MF")

    def getUsersRating(self, users_emb, items_emb):
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos_items, neg_items, users_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        return loss, 0.0

    def forward(self, users, pos_items, neg_items):
        users = users.long()
        pos_items = pos_items.long()
        neg_items = neg_items.long()
        users_emb = self.embedding_user(users)
        pos_emb = self.embedding_item(pos_items)
        neg_emb = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb

class MLP(nn.Module):
    def __init__(self, config, dataset):
        super(MLP, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.emb = config.emb
        self.nlayer = config.layer
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embed_user_MLP = nn.Embedding(self.num_users, self.emb * (2 ** (self.nlayer - 1)))
        self.embed_item_MLP = nn.Embedding(self.num_items, self.emb * (2 ** (self.nlayer - 1)))
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        MLP_modules = []
        for i in range(self.nlayer):
            input_size = self.emb * (2 ** (self.nlayer - i))
            MLP_modules.append(nn.Dropout(p=0.5))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(self.emb, 1)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
    def compare(self, users_emb, items_emb):
        interaction = torch.cat((users_emb, items_emb), -1)
        output_MLP = self.MLP_layers(interaction)
        concat = output_MLP
        scores = self.predict_layer(concat)
        return scores

    def getUsersRating(self, users_emb, items_emb):
        scores = self.compare(users_emb, items_emb)
        return self.f(scores)

    def bpr_loss(self, users, pos, neg, users_emb, pos_emb, neg_emb):
        pos_scores = self.compare(users_emb, pos_emb)
        neg_scores = self.compare(users_emb, neg_emb)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, pos_items, neg_items):
        embed_user = self.embed_user_MLP(users)
        embed_positem = self.embed_item_MLP(pos_items)
        embed_negitem = self.embed_item_MLP(neg_items)
        return embed_user, embed_positem, embed_negitem

class NeuMF(nn.Module):
    def __init__(self, config, dataset):
        super(NeuMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.emb = config.emb
        self.nlayer = config.layer
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embed_user_GMF = nn.Embedding(self.num_users, self.emb)
        self.embed_item_GMF = nn.Embedding(self.num_items, self.emb)
        self.embed_user_MLP = nn.Embedding(self.num_users, self.emb * (2 ** (self.nlayer - 1)))
        self.embed_item_MLP = nn.Embedding(self.num_items, self.emb * (2 ** (self.nlayer - 1)))
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        MLP_modules = []
        for i in range(self.nlayer):
            input_size = self.emb * (2 ** (self.nlayer - i))
            MLP_modules.append(nn.Dropout(p=0.5))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        predict_size = self.emb * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
    def compare(self, users, items):
        embed_user_GMF = self.embed_user_GMF(users)
        embed_item_GMF = self.embed_item_GMF(items)
        output_GMF = embed_user_GMF * embed_item_GMF
        embed_user_MLP = self.embed_user_MLP(users)
        embed_item_MLP = self.embed_item_MLP(items)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction

    def getUsersRating(self, users, items):
        scores = self.compare(users, items)
        return self.f(scores)

    def bpr_loss(self, users, pos_items, neg_items, users_emb, pos_emb, neg_emb):
        pos_scores = self.compare(users, pos_items)
        neg_scores = self.compare(users, neg_items)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        return loss, 0.0

    def forward(self, users, pos_items, neg_items):
        return users, pos_items, neg_items

class NGCF(nn.Module):
    def __init__(self, config, dataset):
        super(NGCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.n_user = self.dataset.n_users
        self.n_item = self.dataset.m_items
        _, self.Graph = self.dataset.getSparseGraph()
        self.emb_size = self.config.emb
        self.layers = self.config.layer
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item, self.emb_size)))
        })

        self.weight_dict = nn.ParameterDict()
        for k in range(self.layers):
            self.weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(self.emb_size, self.emb_size)))})
            self.weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, self.emb_size)))})
            self.weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(self.emb_size, self.emb_size)))})
            self.weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, self.emb_size)))})

    def getUsersRating(self, users_emb, items_emb):
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos_items, neg_items, users_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        loss = torch.mean(nn.functional.sigmoid(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (torch.norm(users_emb, p=2).pow(2) +
                              torch.norm(pos_emb, p=2).pow(2) +
                              torch.norm(neg_emb, p=2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, pos_items, neg_items):
        A_hat = self.Graph

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            ego_embeddings = nn.Dropout(0.1)(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        # all_embeddings = ego_embeddings
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        """
        *********************************************************
        look up.
        """
        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config.emb
        self.n_layers = self.config.layer
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        print('use NORMAL distribution initilizer')
        self.f = nn.Sigmoid()
        _, self.Graph = self.dataset.getSparseGraph()

    def getUsersRating(self, users_emb, items_emb):
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        # light_out = all_emb
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def bpr_loss(self, users, pos_items, neg_items, users_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (torch.norm(self.embedding_user(users), p=2).pow(2) +
                              torch.norm(self.embedding_item(pos_items), p=2).pow(2) +
                              torch.norm(self.embedding_item(neg_items), p=2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, pos_items, neg_items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_items_emb = all_items[pos_items]
        neg_items_emb = all_items[neg_items]
        return users_emb, pos_items_emb, neg_items_emb

class LaGCF(nn.Module):
    def __init__(self, config, dataset):
        super(LaGCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.G_L, self.G_S = self.dataset.getSparseGraph()
        self.emb_dim = self.config.emb
        self.nlayers = self.config.layer
        self.alpha = self.config.alpha
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.emb_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.emb_dim)
        self.w = nn.ParameterDict()
        self.w_dim = 1
        if self.config.IM == 'PN':
            self.w_dim = 1
        elif self.config.IM == 'PM':
            self.w_dim = self.emb_dim

        if self.config.initilizer == 'xavier':
            nn.init.xavier_uniform_(self.embedding_user.weight)
            nn.init.xavier_uniform_(self.embedding_item.weight)
            for k in range(self.nlayers):
                self.w.update({'W_%d' % k: nn.Parameter(nn.init.normal_(torch.FloatTensor(self.w_dim, self.w_dim)))})
            print('use xavier initilizer')
        elif self.config.initilizer == 'normal':
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            for k in range(self.nlayers):
                self.w.update({'W_%d' % k: nn.Parameter(nn.init.normal_(torch.FloatTensor(self.w_dim, self.w_dim)))})
            print('use normal initilizer')
        elif self.config.initilizer == 'pretrain':
            emb_user = np.load('data/' + self.config.dataset + '/embedding_user.npy')
            emb_item = np.load('data/' + self.config.dataset + '/embedding_item.npy')
            self.embedding_user.weight.data.copy_(torch.from_numpy(emb_user))
            self.embedding_item.weight.data.copy_(torch.from_numpy(emb_item))
            for k in range(self.nlayers):
                self.w.update({'W_%d' % k: nn.Parameter(nn.init.normal_(torch.FloatTensor(self.w_dim, self.w_dim)))})
            print('use pretarined embedding')
        self.f = nn.Sigmoid()

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        emb = torch.cat([users_emb, items_emb])
        embs = [emb]

        for layer in range(self.nlayers):
            theta = math.log(self.alpha / (layer + 1) + 1)
            if self.config.AGG == 'L':
                emb = torch.sparse.mm(self.G_L, emb)
            elif self.config.AGG == 'S':
                emb = torch.sparse.mm(self.G_S, emb)
            elif self.config.AGG == 'LS':
                emb = 1/2 * (torch.sparse.mm(self.G_L, emb) + torch.sparse.mm(self.G_S, emb))

            if self.config.LP == 'ZA':
                emb = emb + embs[0]
            elif self.config.LP == 'FA' and layer > 0:
                emb = emb + embs[1]

            if self.config.IM == 'PN':
                emb = theta * emb * self.w['W_%d' % layer] + (1 - theta) * emb
            elif self.config.IM == 'PM':
                emb = theta * torch.mm(emb, self.w['W_%d' % layer]) + (1 - theta) * emb
            embs.append(emb)
        # light_out = emb
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users_emb, items_emb):
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def _Dropnode(self, adj, ratio):
        size = adj.size()
        index = adj.coalesce().indices().t()
        values = adj.coalesce().values()
        random_index = torch.rand(len(values)) + ratio
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / ratio
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def bpr_loss(self, users, pos_items, neg_items, users_emb, pos_emb, neg_emb):
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        if self.config.norm == 'l2':
            p = 2
        elif self.config.norm == 'l1':
            p = 1

        reg_loss = (1 / 2) * (torch.norm(self.embedding_user(users.long()), p=p).pow(p) +
                              torch.norm(self.embedding_item(pos_items.long()), p=p).pow(p) +
                              torch.norm(self.embedding_item(neg_items.long()), p=p).pow(p)) / float(len(users_emb))

        weight_loss = 0.0
        for name, w in self.named_parameters():
            if 'W_' in name:
                weight_loss += torch.norm(w, p=p).pow(p)
        weight_loss = 1/2 * weight_loss

        return loss, reg_loss + weight_loss

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_items_emb = all_items[pos_items]
        neg_items_emb = all_items[neg_items]
        return users_emb, pos_items_emb, neg_items_emb