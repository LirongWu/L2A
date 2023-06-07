import pyro

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphSSL import DistanceCluster, ContextLabel

class GAug(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, temp, num_hop, beta, 
        adj_orig, adj_norm, labels, train_mask, clu_nclass, iter_step, k_hop, model, dataset):

        super(GAug, self).__init__()

        self.model = model
        # self.alpha = alpha

        self.EdgePredictor = EdgePredictor(nfeat, nhid, dataset)
        self.EdgeSampler = EdgeSampler(alpha, temp, model)
        self.EdgeLearning = EdgeLearning(nfeat, nhid, dropout, model)

        if model == 'GCN':
            self.Classifier = GCN_Classifier(nfeat, nhid, nclass, dropout)
        elif model == 'SAGE':
            self.Classifier = Sage_Classifier(nfeat, nhid, nclass, dropout)
        elif model == 'GAT':
            self.Classifier = GAT_Classifier(nfeat, nhid, nclass, dropout)
        else:
            self.Classifier = Classifier(nfeat, nhid, nclass, dropout, num_hop, beta)

        self.ContextLabel = ContextLabel(adj_norm, labels, train_mask, nclass, iter_step, k_hop)
        self.DistanceCluster = DistanceCluster(nhid, adj_orig, clu_nclass)
        
    def forward(self, adj_norm, adj_orig, features):

        adj_logits = self.EdgePredictor(adj_norm, features)
        # if self.alpha == 0.0:
        #     adj_sampled = adj_norm
        # else:
        #     adj_sampled = self.EdgeSampler(adj_logits, adj_orig)
        adj_sampled = self.EdgeSampler(adj_logits, adj_orig)

        adj_new, embedding = self.EdgeLearning(adj_sampled, features)
        nc_logits = self.Classifier(adj_new, features)

        return nc_logits, adj_logits, adj_sampled, embedding

        
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None, dropout=0.0, bias=True):
        super(GCNLayer, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.activation = activation
        self.dropout = dropout

    def forward(self, adj, x):
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.linear(x)
        h = adj @ h

        if self.activation:
            h = self.activation(h)

        return h


class SageConv(nn.Module):

    def __init__(self, input_dim, output_dim, bias=True):
        super(SageConv, self).__init__()

        self.linear = nn.Linear(input_dim*2, output_dim, bias=bias)

    def forward(self, adj, x):
        neigh_x = adj @ x
        concat_x = torch.cat([neigh_x, x], dim=-1)
        h = self.linear(concat_x)

        return h


class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None, dropout=0.0, nhead=4, bias=True):

        super(GATLayer, self).__init__()
        self.nhead = nhead
        self.dropout = dropout
        self.activation = activation

        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.attn_l = nn.Linear(output_dim, self.nhead, bias=False)
        self.attn_r = nn.Linear(output_dim, self.nhead, bias=False)
        
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None

        self.init_params()

    def init_params(self):

        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, x):

        h = x @ self.W

        el = self.attn_l(h)
        er = self.attn_r(h)

        edge_list = adj.nonzero().T
        attn = el[edge_list[0]] + er[edge_list[1]]
        attn = torch.exp(F.leaky_relu(attn, negative_slope=0.2).squeeze())

        if self.nhead == 1:
            adj_new = torch.zeros(size=(adj.shape[0], adj.shape[1]), device=adj.device)
            adj_new.index_put_((edge_list[0], edge_list[1]), attn)
        else:
            adj_new = torch.zeros(size=(adj.shape[0], adj.shape[1], self.nhead), device=adj.device)
            adj_new.index_put_((edge_list[0], edge_list[1]), attn)
            adj_new.transpose_(1, 2)

        adj_new = F.normalize(adj_new, p=1, dim=-1)
        adj_new = F.dropout(adj_new, self.dropout, training=self.training)

        h = adj_new @ h

        if self.b is not None:
            h = h + self.b
        if self.activation:
            h = self.activation(h)
        if self.nhead > 1:
            h = h.flatten(start_dim=1)

        return h


# Parameterized Augmentation Distribution
class EdgePredictor(nn.Module):
    def __init__(self, nfeat, nhid, dataset):
        super(EdgePredictor, self).__init__()

        self.dataset = dataset
        self.gcn1 = GCNLayer(nfeat, nhid, None, 0, bias=False)
        self.gcn2 = GCNLayer(nhid, nhid, F.relu, 0, bias=False)

    def forward(self, adj, x):

        if self.dataset != 'cora' and self.dataset != 'citeseer':
            x = torch.nn.functional.normalize(x, p=1, dim=1)
        h = self.gcn1(adj, x)
        h = self.gcn2(adj, h)
        adj_logits = h @ h.T

        return adj_logits


# Gumbel-Softmax Sampling
class EdgeSampler(nn.Module):
    def __init__(self, alpha, temp, model):
        super(EdgeSampler, self).__init__()

        self.alpha = alpha
        self.temp = temp
        self.model = model

    def forward(self, adj_logits, adj_orig):

        edge_probs = adj_logits / torch.max(adj_logits)
        edge_probs = self.alpha * edge_probs + (1-self.alpha) * adj_orig

        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temp, probs=edge_probs).rsample()
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T

        adj_sampled.fill_diagonal_(1)
        D_norm = torch.diag(torch.pow(adj_sampled.sum(1), -0.5))
        adj_sampled = D_norm @ adj_sampled @ D_norm

        return adj_sampled


# Learning Weighted Graph
class EdgeLearning(nn.Module):
    def __init__(self, nfeat, nhid, dropout, model):
        super(EdgeLearning, self).__init__()

        self.model = model
        self.dropout = dropout
        self.linear = nn.Linear(nfeat, nhid)
        self.att = nn.Linear(nhid*2, 1, bias=False)

    def forward(self, adj, x):

        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.linear(x)

        edge_list = adj.nonzero().t()
        edge_h = torch.cat((h[edge_list[0, :], :], h[edge_list[1, :], :]), dim=1)

        e = torch.tanh(self.att(edge_h).squeeze())
        new_adj = torch.sparse.FloatTensor(edge_list, e, torch.Size([h.shape[0], h.shape[0]])).to_dense()

        return new_adj * adj, h


# GNN Classifier
class Classifier(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_hop, beta):
        super(Classifier, self).__init__()

        self.beta = beta
        self.num_hop = num_hop
        self.dropout = dropout

        self.linear1 = nn.Linear(nfeat, nhid)
        self.linear2 = nn.Linear(nhid, nclass)

    def forward(self, adj, x):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        h = x

        for _ in range(self.num_hop):
            h = adj @ h
            h = h * (1.0 - self.beta) + x * self.beta

        h = self.linear2(h)

        return h


class GCN_Classifier(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gcn1 = GCNLayer(nfeat, nhid)
        self.gcn2 = GCNLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gcn1(adj, x))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn2(adj, x)

        return x


class Sage_Classifier(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Sage_Classifier, self).__init__()

        self.sage1 = SageConv(nfeat, nhid)
        self.sage2 = SageConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage1(adj, x))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage2(adj, x)

        return x


class GAT_Classifier(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, dropout, nheads=1):
        super(GAT_Classifier, self).__init__()

        self.gat1 = GATLayer(nfeat, int(nhid / nheads), activation=F.elu, dropout=dropout, nhead=nheads)
        self.gat2 = GATLayer(nhid, nclass, activation=None, dropout=dropout, nhead=1)
        self.dropout = dropout

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat1(adj, x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(adj, x)

        return x