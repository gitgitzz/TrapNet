
import torch
import numpy as np
import os
import json
import tqdm
import networkx as nx
import time



def gaussian_kernel(x1, x2, kernel_mul=2.0, kernel_num=5, fix_sigma=0, mean_sigma=0):
    """
    :param x1:
    :param x2:
    :param kernel_mul:
    :param kernel_num:
    :param fix_sigma:
    :param mean_sigma:
    if dont assign fix_sigma AND not use mean_sigma, Defaults to use median distance as sigma
    - Maximum Mean Discrepancy: https://docs.seldon.io/projects/alibi-detect/en/stable/methods/mmddrift.html
    :return: Gram matrix
    """
    x1_sample_size = x1.shape[0]
    x2_sample_size = x2.shape[0]
    x1_tile_shape = []
    x2_tile_shape = []
    norm_shape = []
    for i in range(len(x1.shape) + 1):
        if i == 1:
            x1_tile_shape.append(x2_sample_size)
        else:
            x1_tile_shape.append(1)
        if i == 0:
            x2_tile_shape.append(x1_sample_size)
        else:
            x2_tile_shape.append(1)
        if not (i == 0 or i == 1):
            norm_shape.append(i)

    tile_x1 = torch.unsqueeze(x1, 1).repeat(x1_tile_shape)
    tile_x2 = torch.unsqueeze(x2, 0).repeat(x2_tile_shape)
    L2_distance = torch.square(tile_x1 - tile_x2).sum(dim=norm_shape)
    # print(L2_distance)
    # bandwidth inference
    if fix_sigma:
        bandwidth = fix_sigma
    elif mean_sigma:
        bandwidth = torch.mean(L2_distance)
    else:  ## median distance
        bandwidth = torch.median(L2_distance.reshape(L2_distance.shape[0],-1))
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # print(bandwidth_list)
    #print(torch.cat(bandwidth_list,0).to(torch.device('cpu')).numpy())
    ## gaussian_RBF = exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val), L2_distance


if __name__ == '__main__':
    num_classes = 1000
    path = "./private_feats/celeba"
    q_min, q_max = 0.00, 0.25
    # max_matching(path, num_classes, q_min, q_max)

    true_feat = torch.from_numpy(np.load(os.path.join(path, "private_feats.npy"))).float()
    info = torch.from_numpy(np.load(os.path.join(path, "private_targets.npy"))).view(-1).long()

    # calc the mean feature of each class
    feat_cls = None
    for i in range(0, num_classes):
        index = info == i
        if i == 0:
            feat_cls = torch.mean(true_feat[index, :], dim=0, keepdim=True)
        else:
            feat_cls = torch.cat([feat_cls, torch.mean(true_feat[index, :], dim=0, keepdim=True)], dim=0)

    # dist_matrix between each class
    _, l2_matrix = gaussian_kernel(feat_cls, feat_cls)
    # calc adjacent matrix based on predefined threshold in q
    q = torch.tensor([q_min, q_max])
    q_value = torch.quantile(l2_matrix.view(-1,), q, dim=0, keepdim=True)
    adjacent_matrix = torch.where((q_value[0].item() < l2_matrix) & (l2_matrix < q_value[-1].item()), l2_matrix, l2_matrix[0,0])

    G = nx.from_numpy_array(l2_matrix.numpy())
    t1 = time.time()
    match_results = nx.min_weight_matching(G, weight='weight')
    interval = time.time() - t1
    print("Time:{:2f}".format(interval))
    node_matches = {}
    for res in sorted(match_results):
        node_matches[int(res[0])] = int(res[1])
        node_matches[int(res[1])] = int(res[0])
    # save node matching results
    json_path = "celeba_node_matches_min.txt"
    with open(json_path, "w") as fp:
        json.dump(node_matches, fp)



