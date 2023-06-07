import os
import csv
import nni
import time
import json
import argparse
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from utils import *
from model import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Pretrain Graph Augmentor
def pretrain_EdgePredictor(model):

    optimizer = torch.optim.Adam(model.EdgePredictor.parameters(), lr=param['lr'])
    
    model.train()
    for epoch in range(param['ep_epochs']):

        adj_logits = model.EdgePredictor(adj_norm, features)
        ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
        cl_loss = model.ContextLabel(model.EdgeSampler(adj_logits, adj_orig))
        if param['ratio_cl']:
            loss = param['ratio_cl'] * cl_loss + ep_loss
        else:
            loss = ep_loss
            cl_loss = param['ratio_cl'] * ep_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\033[0;30;43m Pretrain EdgePredictor, Epoch [{}/{}]: EP Loss {:.6f}, CL Loss {:.6f} \033[0m'.format(epoch+1, param['ep_epochs'], ep_loss.item(), cl_loss.item()))


# Pretrain GNN Classifier
def pretrain_Classifier(model):

    optimizer_edge = torch.optim.Adam(list(model.EdgeLearning.parameters())+list(model.DistanceCluster.parameters()), lr=param['lr'], weight_decay=param['weight_decay'])
    optimizer_cla = torch.optim.Adam(model.Classifier.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])

    if len(labels.size()) == 2:
        nc_criterion = nn.BCEWithLogitsLoss()
    else:
        nc_criterion = nn.CrossEntropyLoss()

    es = 0
    test_best = 0
    test_val = 0
    val_best = 0

    for epoch in range(param['nc_epochs']):
        model.train()

        adj_new, embedding = model.EdgeLearning(adj_norm, features)
        nc_logits = model.Classifier(adj_new, features)
        nc_loss = nc_criterion(nc_logits[train_mask], labels[train_mask])
        dc_loss = model.DistanceCluster(embedding)
        loss = param['ratio_dc'] * dc_loss + nc_loss

        optimizer_edge.zero_grad()
        optimizer_cla.zero_grad()
        loss.backward()
        optimizer_edge.step()
        optimizer_cla.step()
        
        model.eval()
        adj_new, _ = model.EdgeLearning(adj_norm, features)
        nc_logits_eval = model.Classifier(adj_new, features)
        train_acc = evaluation(nc_logits_eval[train_mask], labels[train_mask])
        val_acc = evaluation(nc_logits_eval[val_mask], labels[val_mask])
        test_acc = evaluation(nc_logits_eval[test_mask], labels[test_mask])

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            es = 0
        else:
            es += 1
            if es == 1000:
                print('Early stop!')
                break

        print('\033[0;30;41m Pretrain Classifier, Epoch [{}/{}]: NC Loss {:.6f}, DC Loss {:.6f}, Train Acc {:.4f}, Val Acc {:.4f}, Test Acc {:.4f} | {:.4f}, {:.4f}\033[0m'.format(
            epoch+1, param['nc_epochs'], nc_loss.item(), dc_loss.item(), train_acc, val_acc, test_acc, test_val, test_best))


def main():
    
    model = GAug(nfeat=features.shape[1], nhid=param['nhid'], nclass=nclass, dropout=param['dropout'], alpha=param['alpha'], temp=param['temp'], 
            num_hop=param['num_hop'], beta=param['beta'], adj_orig=adj_orig, adj_norm=adj_norm, labels=labels, train_mask=train_mask,  
            clu_nclass=param['clu_nclass'], iter_step=param['iter_step'], k_hop=param['k_hop'], model=param['model'], dataset=param['dataset']).to(device)

    if param['ep_epochs']:
        pretrain_EdgePredictor(model)
    if param['nc_epochs']:
        pretrain_Classifier(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
   
    if len(labels.size()) == 2:
        nc_criterion = nn.BCEWithLogitsLoss()
    else:
        nc_criterion = nn.CrossEntropyLoss()

    es = 0
    test_best = 0
    test_val = 0
    val_best = 0

    for epoch in range(param['n_epochs']):

        model.train()

        nc_logits, adj_logits, adj_sampled, embedding = model(adj_norm, adj_orig, features)
        nc_loss = nc_criterion(nc_logits[train_mask], labels[train_mask])
        ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
        dc_loss = model.DistanceCluster(embedding)
        cl_loss = model.ContextLabel(adj_sampled)
        if param['ratio_cl']:
            loss = param['ratio_ep'] * ep_loss + param['ratio_dc'] * dc_loss + param['ratio_cl'] * cl_loss + nc_loss
        else:
            loss = param['ratio_ep'] * ep_loss + param['ratio_dc'] * dc_loss + nc_loss
            cl_loss = param['ratio_cl'] * ep_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        adj_new, _ = model.EdgeLearning(adj_norm, features)
        nc_logits_eval = model.Classifier(adj_new, features)
        train_acc = evaluation(nc_logits_eval[train_mask], labels[train_mask])
        val_acc = evaluation(nc_logits_eval[val_mask], labels[val_mask])
        test_acc = evaluation(nc_logits_eval[test_mask], labels[test_mask])

        if test_acc > test_best:
            test_best = test_acc

        if val_acc > val_best:
            val_best = val_acc
            test_val = test_acc
            es = 0
        else:
            es += 1
            if es == 200:
                print('Early stop!')
                break

        print('\033[0;30;46m Epoch [{:3}/{}]: EP Loss {:.6f}, CL Loss {:.6f}, NC Loss {:.6f}, DC Loss {:.6f}, Train Acc {:.4f}, Val Acc {:.4f}, Test Acc {:.4f} | {:.4f}, {:.4f}\033[0m'.format(
            epoch+1, param['n_epochs'], ep_loss.item(), cl_loss.item(), nc_loss.item(), dc_loss.item(), train_acc, val_acc, test_acc, test_val, test_best))

    return test_acc, test_val, test_best

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'blogcatalog', 'cornell', 'texas', 'wisconsin', 'film'])

    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=0.8)
    parser.add_argument('--num_hop', type=int, default=5)
    parser.add_argument('--beta', type=int, default=0.1)
    parser.add_argument('--ratio_ep', type=float, default=0.5)
    parser.add_argument('--ratio_dc', type=float, default=0.1)
    parser.add_argument('--ratio_cl', type=float, default=1.0)
    parser.add_argument('--iter_step', type=int, default=100)
    parser.add_argument('--clu_nclass', type=int, default=20)
    parser.add_argument('--k_hop', type=int, default=1)

    parser.add_argument('--model', type=str, default='SEM')    
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--ep_epochs', type=int, default=200)
    parser.add_argument('--nc_epochs', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_mode', type=int, default=0)
    parser.add_argument('--data_mode', type=int, default=0)
    parser.add_argument('--model_mode', type=int, default=0)

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())

    if os.path.exists("../param/best_parameters.json"):
        param = json.loads(open("../param/best_parameters.json", 'r').read())[param['dataset']][param['model']]

     # Load Dataset
    features, adj_orig, adj_norm, labels, train_mask, val_mask, test_mask, nclass = load_data(param['dataset'])
    norm_w = adj_orig.shape[0]**2 / float((adj_orig.shape[0]**2 - adj_orig.sum()) * 2)
    pos_weight = torch.FloatTensor([float(adj_orig.shape[0]**2 - adj_orig.sum()) / adj_orig.sum()]).to(device)

    if param['clu_nclass'] == 0:
        param['clu_nclass'] = 3 * nclass

    if param['save_mode'] == 0:
        SetSeed(param['seed'])
        test_acc, test_val, test_best = main()
        nni.report_final_result(test_val)

    else:
        test_acc_list = []
        test_val_list = []
        test_best_list = []

        for seed in range(5):
            SetSeed(seed + param['seed'] * 5)
            test_acc, test_val, test_best = main()
            test_acc_list.append(test_acc)
            test_val_list.append(test_val)
            test_best_list.append(test_best)
            nni.report_intermediate_result(test_val)

        nni.report_final_result(np.mean(test_val_list))
        outFile = open('../PerformMetrics.csv','a+', newline='')
        writer = csv.writer(outFile, dialect='excel')
        results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
        for v, k in param.items():
            results.append(k)
        
        results.append(str(test_acc_list))
        results.append(str(test_val_list))
        results.append(str(test_best_list))
        results.append(str(np.mean(test_acc_list)))
        results.append(str(np.mean(test_val_list)))
        results.append(str(np.mean(test_best_list)))
        results.append(str(np.std(test_acc_list)))
        results.append(str(np.std(test_val_list)))
        results.append(str(np.std(test_best_list)))
        writer.writerow(results)

