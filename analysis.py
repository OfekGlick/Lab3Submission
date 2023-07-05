import pandas as pd
import requests
import os
from torch_geometric.data import Dataset
import torch
from torch.nn import Linear
from torch.nn import functional as f
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, GATv2Conv
import networkx as nx
from torch_geometric.utils import degree, to_networkx
# import community
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

CONV_TYPES = {'GCN': GCNConv, 'SAGE': SAGEConv, 'CHEB': ChebConv, 'GAT': GATv2Conv}
from dataset import HW3Dataset, GCN

# dataset = HW3Dataset(root='data/hw3/')
# graph = dataset[0]
# tsne = TSNE(n_components=3, n_iter=250).fit_transform(graph.x)
# print("tsne - done")
# pca = PCA(n_components=2).fit(graph.x)
# print("pca - done")
#
# num_nodes = graph.num_nodes
# num_edges = graph.num_edges
# average_degree = degree(graph.edge_index[0]).mean().item()
# degree_lst = list([int(bla.item()) for bla in degree(graph.edge_index[0])])
#
# plt.hist(degree_lst, log=True, density=True, bins=25)
# plt.xlabel("Node Degree")
# plt.ylabel("Frequency (log-scaled)")
# plt.show()
#
# year_lst = list([int(bla.item()) for bla in graph.node_year])
# plt.hist(year_lst, log=True, density=True, bins=25)
# plt.xlabel("Publishing Year")
# plt.ylabel("Frequency (log-scaled)")
# plt.show()
#
# from collections import Counter
# categories = list([int(bla.item()) for bla in graph.y])
# bla = dict(Counter(categories))
# fig, ax = plt.subplots()
# ax.bar(bla.values(), labels=bla.keys())
# plt.show()
# ofek = 5
# print(f"Number of nodes: {num_nodes}")
# print(f"Number of edges: {num_edges}")
# print(f"Average degree: {average_degree}")
#
#
#
#
#
# # G = to_networkx(graph)
# # degree_centrality = nx.degree_centrality(G)
# # betweenness_centrality = nx.betweenness_centrality(G)
# # eigenvector_centrality = nx.eigenvector_centrality(G)
# #
# # # Print centrality measures for a few nodes
# # print("Node Centrality Measures:")
# # for node in [0, 1, 2]:
# #     print(f"Node {node}:")
# #     print(f"Degree Centrality: {degree_centrality[node]}")
# #     print(f"Betweenness Centrality: {betweenness_centrality[node]}")
# #     print(f"Eigenvector Centrality: {eigenvector_centrality[node]}")
# #     print()
# import pandas as pd
# import matplotlib.pyplot
#
# df = pd.read_csv("tsne.csv")
# from collections import Counter
#
# temp = dict(Counter(df.label))
# top5 = list(sorted([(label, count) for label, count in temp.items()], key=lambda x: x[1], reverse=True))[:5]
# top5 = [bla for bla, _ in top5]
# last5 = list(sorted([(label, count) for label, count in temp.items()], key=lambda x: x[1], reverse=True))[-5:]
# last5 = [bla for bla, _ in last5]
# df_top_5 = df[(df.x < 2) & (df.x > -2) & (df.y < 2) & (df.y > -2) & (df.label.isin(top5))]
# df_last_5 = df[(df.x < 2) & (df.x > -2) & (df.y < 2) & (df.y > -2) & (df.label.isin(last5))]
# plt.scatter(x=df_last_5.x, y=df_last_5.y, c=df_last_5.label)
# plt.title("T-SNE representation of Skip-Gram - Least Common labels")
# plt.show()
# plt.scatter(x=df_top_5.x, y=df_top_5.y, c=df_top_5.label)
# plt.title("T-SNE representation of Skip-Gram - Most Common labels")
# plt.show()
dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = {'n_layers': 4,
               'channel_increase': 8,
               'dropout': 0.22292,
               'activation': 'tanh',
               'conv_type': 'GCN'}

model = GCN(config_file).to(device)
model.load_state_dict(torch.load('best_model_best_config.pth'))
model.eval()
pred = model(data.x.to(device), data.edge_index.to(device))

pred = pred.argmax(dim=1).to(device)


def calc_stats(pred, gt):
    from sklearn.metrics import precision_recall_fscore_support
    from collections import Counter
    acc = (pred.eq(gt.to(device).squeeze()).sum().item()) / len(gt)
    predictions = pred.cpu().numpy()
    labels = gt.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions)
    categories = list([int(bla.item()) for bla in data.y[data.val_mask]])
    label_count = dict(Counter(categories))
    label_count = {k: {'ratio_from_val': v / sum(label_count.values()),
                       'precision': precision[k],
                       'recall': recall[k],
                       'f1': f1[k]} for k, v in label_count.items()}
    fig, ax = plt.subplots()
    x = [values['ratio_from_val'] for values in label_count.values()]
    y_precision = [values['precision'] for values in label_count.values()]
    y_recall = [values['recall'] for values in label_count.values()]
    y_f1 = [values['f1'] for values in label_count.values()]
    plt.scatter(x, y_f1, label='F1', c='cyan')
    plt.xlabel("Ratio of label in the validation set")
    plt.title("F1 score of different labels")
    plt.show()
    plt.scatter(x, y_precision, label='Precision', c='violet')
    plt.xlabel("Ratio of label in the validation set")
    plt.title("Precision score of different labels")
    plt.show()
    plt.scatter(x, y_recall, label='Recall', c='pink')
    plt.xlabel("Ratio of label in the validation set")
    plt.title("Recall score of different labels")
    plt.show()


    # Set labels and title
    ax.set_xlabel('ratio_from_val')
    ax.set_ylabel('Values')
    ax.set_title('Graph with Multiple Lines')

    # Add legend
    ax.legend()
    ofek = 5
    # Display the graph
    plt.show()
    print()
    return acc, precision, recall, f1




# pred_train = pred[data.val_mask.to(device)]
# gt_train = data.y.to(device)[data.val_mask.to(device)]
# calc_stats(pred_train, gt_train)
# pred_val = pred[data.val_mask.to(device)]
# gt_val = data.y.to(device)[data.val_mask.to(device)]
# calc_stats(pred_val, gt_val)


df = pd.read_csv('prediction.csv')
ofek = 5
