# %%
import requests
import os
from torch_geometric.data import Dataset
import torch
from torch.nn import Linear
from torch.nn import functional as f
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, GATv2Conv
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

CONV_TYPES = {'GCN': GCNConv, 'SAGE': SAGEConv, 'CHEB': ChebConv, 'GAT': GATv2Conv}

# set seed:
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


class HW3Dataset(Dataset):
    url = 'https://technionmail-my.sharepoint.com/:u:/g/personal/ploznik_campus_technion_ac_il/EUHUDSoVnitIrEA6ALsAK1QBpphP5jX3OmGyZAgnbUFo0A?download=1'

    def __init__(self, root, transform=None, pre_transform=None):
        super(HW3Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        file_url = self.url.replace(' ', '%20')
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download the file, status code: {response.status_code}")

        with open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'wb') as f:
            f.write(response.content)

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(raw_path)
        torch.save(data, self.processed_paths[0])

    def len(self):
        return 1

    def get(self, idx):
        return torch.load(self.processed_paths[0])


class GCN(torch.nn.Module):
    def __init__(self, config, num_classes, num_features):
        super().__init__()
        # Assigning parameters from wandb sweep config
        self.n_layers = config['n_layers']
        self.channel_increase = config['channel_increase']
        self.dropout = config['dropout']
        self.activation = config['activation']
        self.conv_type = config['conv_type']
        layer_dims = [self.channel_increase ** i for i in range(2, self.n_layers // 2 + 2)]
        layer_dims = layer_dims + layer_dims[::-1]
        # Use self.n_layers to initialize conv layers dynamically
        self.convs = torch.nn.ModuleList()
        if self.conv_type == 'GAT':
            self.convs.append(GATv2Conv(num_features, layer_dims[0], 4))
        elif self.conv_type == 'CHEB':
            self.convs.append(CONV_TYPES[self.conv_type](num_features, layer_dims[0], K=2))
        else:
            self.convs.append(CONV_TYPES[self.conv_type](num_features, layer_dims[0]))
        for i in range(self.n_layers - 1):
            if self.conv_type == 'GAT':
                if i < self.n_layers - 2:
                    self.convs.append(GATv2Conv(4 * layer_dims[i], layer_dims[i + 1], 4))
                else:
                    self.convs.append(GATv2Conv(4 * layer_dims[i], layer_dims[i + 1], 1))
            elif self.conv_type == 'CHEB':
                self.convs.append(CONV_TYPES[self.conv_type](layer_dims[i], layer_dims[i + 1], K=2))
            else:
                self.convs.append(CONV_TYPES[self.conv_type](layer_dims[i], layer_dims[i + 1]))
        self.classifier = Linear(layer_dims[-1], num_classes)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            if self.activation == 'tanh':
                x = torch.tanh(x)
            elif self.activation == 'relu':
                x = torch.relu(x)
            elif self.activation == 'leaky_relu':
                x = f.leaky_relu(x)
            # Add more activations as required
            x = f.dropout(x, training=self.training, p=self.dropout)
        out = self.classifier(x)
        return out


def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None, file_name='i'):
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.savefig(f"scatter_{file_name}.png")


if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = {'n_layers': 4,
                   'channel_increase': 8,
                   'dropout': 0.22292,
                   'activation': 'tanh',
                   'conv_type': 'GCN'}
    lr = config_file['lr']
    model = GCN(config_file).to(device)

    print(model)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in (pbar := tqdm(range(5000))):
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask].squeeze(-1))
        test_loss = criterion(out[data.val_mask], data.y[data.val_mask].squeeze())
        train_loss.backward()
        optimizer.step()
    model.eval()
    pred = model(data.x.to(device), data.edge_index.to(device))
    pred = pred.argmax(dim=1)
    train_acc = (pred[data.train_mask].eq(data.y[data.train_mask].squeeze()).sum().item()) / len(
        data.train_mask)
    test_acc = (pred[data.val_mask].eq(data.y[data.val_mask].squeeze()).sum().item()) / len(data.val_mask)
    pbar.set_description(f"Loss: - {train_loss.item(): .4f}, Test Accuracy: {test_acc:.8f}")
