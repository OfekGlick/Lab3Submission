from dataset import HW3Dataset, GCN
import pandas as pd
import torch

if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = {'n_layers': 4,
                   'channel_increase': 8,
                   'dropout': 0.22292,
                   'activation': 'tanh',
                   'conv_type': 'GCN'}
    num_classes = dataset.num_classes
    num_features = dataset.num_features
    model = GCN(config_file, num_classes, num_features).to(device)
    model.load_state_dict(torch.load('best_model_best_config.pth'))
    model.eval()
    pred = model(data.x.to(device), data.edge_index.to(device)).tolist()
    df = pd.DataFrame({'idx': list(range(len(pred))), 'prediction': pred})
    print("Saving prediction.csv file")
    df.to_csv('prediction.csv', index=False)
    print("prediction.csv file saved!")
