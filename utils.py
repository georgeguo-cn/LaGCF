#!/usr/bin/python -u
import torch
from tqdm import tqdm
import parse
import numpy as np
from time import time
from dataloader import BasicDataset
from sklearn.metrics import roc_auc_score

args = parse.parse_args()
topks = eval(args.topks)
try:
    import cppimport, sys, os
    sys.path.append(os.getcwd() + '/sources')
    # print(sys.path)
    sampling = cppimport.imp("sampling")
    sampling.seed(args.seed)
    sample_ext = True
except Exception as e:
    print(e)
    print("Cpp extension not loaded")
    sample_ext = False

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)

def UniformSample_original(dataset, neg_ratio=1):
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items, dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    # print('sample time:', time() - start)
    return S

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', args.bpr_batch)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(config, dataset, Rmodel, epoch, w=None, device='cpu'):
    dataset: BasicDataset
    testDict = dataset.testDict
    users = list(testDict.keys())
    if args.model in ['mlp', 'ncf']:
        u_batch_size = 1
    elif len(users) / 10 > 100:
        u_batch_size = 100
    else:
        u_batch_size = 10
    # eval mode with no dropout
    Rmodel = Rmodel.eval()
    max_K = max(topks)
    results = {'precision': np.zeros(len(topks)), 'recall': np.zeros(len(topks)), 'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        batch_num = len(users) // u_batch_size
        if batch_num * u_batch_size == len(users):
            total_batch = batch_num
        else:
            total_batch = batch_num + 1
        user_batch = torch.LongTensor(range(dataset.n_users)).to(device)
        item_batch = torch.LongTensor(range(dataset.m_items)).to(device)
        users_embeddings, item_embeddings, _ = Rmodel(user_batch, item_batch, item_batch)
        ratings = Rmodel.getUsersRating(users_embeddings, item_embeddings)
        for batch_users in tqdm(minibatch(users, batch_size=u_batch_size), desc='Test'):
            batch_users_cp = batch_users
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            if config.model in ['mlp', 'ncf']:
                batch_users = [batch_users[0] for _ in range(dataset.m_items)]
            rating = ratings[batch_users]
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users_cp)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        # print(rating_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
        # scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        if args.tensorboard:
            rec_dict, pre_dict, ndcg_dict = {}, {}, {}
            for i in range(len(topks)):
                rec_dict[str(topks[i])] = results['recall'][i]
                pre_dict[str(topks[i])] = results['precision'][i]
                ndcg_dict[str(topks[i])] = results['ndcg'][i]
            w.add_scalars(f'Test/Recall@{topks}', rec_dict, epoch)
            w.add_scalars(f'Test/Precision@{topks}', pre_dict, epoch)
            w.add_scalars(f'Test/NDCG@{topks}', ndcg_dict, epoch)
        return results, users_embeddings, item_embeddings

def early_stop(model, epoch, max_epoch, value, max_value, step, flag_step, dataset):
    emb_path = 'data/' + dataset + '/' + 'LF/' + str(args.layer) + '/'
    if value > max_value:
        step = 0
        max_value = value
        max_epoch = epoch
    else:
        step += 1

    if step >= flag_step:
        should_stop = True
    else:
        should_stop = False
    return max_epoch, max_value, step, should_stop

# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset: BasicDataset
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================