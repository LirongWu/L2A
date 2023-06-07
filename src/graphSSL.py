import pymetis
import collections
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ClusteringMachine(object):
    def __init__(self, adj, nclass):

        self.adj = adj.detach().cpu().numpy()
        self.nclass = nclass
        self.graph = nx.from_numpy_matrix(self.adj)
        
    def decompose(self):

        print("Metis graph clustering started ...")
        self.metis_clustering()
        self.central_nodes = self.get_central_nodes()
        self.shortest_path_to_clusters(self.central_nodes)
        self.dis_matrix = torch.FloatTensor(self.dis_matrix)

    def metis_clustering(self):

        (st, parts) = pymetis.part_graph(adjacency=self.graph, nparts=self.nclass)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def general_data_partitioning(self):

        self.sg_nodes = {}
        self.sg_edges = {}

        for cluster in self.clusters:

            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]

            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]

        print('Number of nodes in clusters:', {x: len(y) for x, y in self.sg_nodes.items()})

    def get_central_nodes(self):

        self.general_data_partitioning()
        central_nodes = {}

        for cluster in self.clusters:

            counter = {}
            for node, _ in self.sg_edges[cluster]:
                counter[node] = counter.get(node, 0) + 1

            sorted_counter = sorted(counter.items(), key=lambda x:x[1])
            central_nodes[cluster] = sorted_counter[-1][0]

        return central_nodes

    def shortest_path_to_clusters(self, central_nodes, transform=True):

        self.dis_matrix = -np.ones((self.adj.shape[0], self.nclass))

        for cluster in self.clusters:
            node_cur = central_nodes[cluster]
            visited = set([node_cur])
            q = collections.deque([(x, 1) for x in self.graph.neighbors(node_cur)])

            while q:
                node_cur, depth = q.popleft()

                if node_cur in visited:
                    continue
                visited.add(node_cur)

                if transform:
                    self.dis_matrix[node_cur][cluster] = 1 / depth
                else:
                    self.dis_matrix[node_cur][cluster] = depth
                    
                for node_next in self.graph.neighbors(node_cur):
                    q.append((node_next, depth+1))

        if transform:
            self.dis_matrix[self.dis_matrix==-1] = 0
        else:
            self.dis_matrix[self.dis_matrix==-1] = self.dis_matrix.max() + 2

        return self.dis_matrix
 
# Global-Path Prediction 
class DistanceCluster(nn.Module):
    def __init__(self, nhid, adj, clu_nclass):
        super(DistanceCluster, self).__init__()

        self.linear = nn.Linear(nhid, clu_nclass)
        self.cluster_agent = ClusteringMachine(adj, clu_nclass)
        self.cluster_agent.decompose()
        self.pseudo_labels = self.cluster_agent.dis_matrix.to(device)

    def forward(self, embeddings):
        
        output = self.linear(embeddings)
        loss = F.mse_loss(output, self.pseudo_labels, reduction='mean')

        return loss

# Label Distribution Preservation
class ContextLabel(nn.Module):
    def __init__(self, adj_norm, labels, train_mask, nclass, iter_step, k_hop):
        super(ContextLabel, self).__init__()

        self.train_mask = train_mask
        self.iter_step = iter_step
        self.k_hop = k_hop

        self.labels_onehot = torch.eye(nclass)[labels, :].to(device)
        self.pseudo_labels = self.LabelPropagate(adj_norm, self.labels_onehot).detach()

    def forward(self, adj):
        
        output = self.LabelPropagate(adj, self.labels_onehot)
        loss = F.mse_loss(output, self.pseudo_labels, reduction='mean')

        return loss

    def LabelPropagate(self, adj, labels):

        Y_pred = torch.zeros(labels.shape).to(device)
        Y_pred[self.train_mask] = labels[self.train_mask]

        for _ in range(self.iter_step):
            Y_pred = adj @ Y_pred
            Y_pred[self.train_mask] = labels[self.train_mask]

        if self.k_hop == 0:
            return Y_pred
        else:
            Y_pred = F.gumbel_softmax(Y_pred, hard=True)
            Y_pred[self.train_mask] = labels[self.train_mask]
            Y_label_distribution = self.get_label_distribution(adj, Y_pred)

            return Y_label_distribution

    def get_label_distribution(self, adj_norm, labels):

        adj = adj_norm
        for _ in range(self.k_hop-1):
            adj = adj @ adj_norm
        
        edge_list = adj.nonzero().t()
        new_adj = torch.sparse.FloatTensor(edge_list, labels[edge_list[1]], torch.Size([adj.shape[0], adj.shape[0], labels.shape[1]])).to_dense()

        label_distribution = new_adj.sum(dim=1)
        label_distribution = label_distribution / label_distribution.sum(dim=1, keepdim=True)

        return label_distribution