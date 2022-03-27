import torch_geometric as pyg
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from gcn import Gcn
from gin import Gin
from torch.utils.data import Dataset, DataLoader
import argparse



class GraphData(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return (self.df.x, self.df.edge_index, self.df.y, self.df.train_mask, self.df.val_mask)


df = pyg.datasets.Planetoid("datasets/planetoid", "citeseer")



# trainer = pl.Trainer(gpus=1)
# data = df.data
# # data = (data.x, data.edge_list, )
# data = GraphData(data)
# loader = DataLoader(data, batch_size=1, collate_fn=lambda x: x)
# torch.autograd.set_detect_anomaly(True)
# trainer.fit(model, loader)

def accuracy(probs, y):
    pred = torch.argmax(probs, dim=1)
    bools = torch.eq(pred, y).float()
    return torch.mean(bools) * 100


def train_gcn(num_layers, latent_dim):
    model = Gcn(num_layers, 3703, latent_dim, 7)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fun = torch.nn.CrossEntropyLoss()

    for ep in tqdm(range(100)):
        model.train()
        optim.zero_grad()
        out = model(df.data.x, df.data.edge_index.T)
        train_mask, val_mask = df.data.train_mask, df.data.val_mask
        loss = loss_fun(out[train_mask], df.data.y[train_mask])
        loss.backward()

        print("Epoch {}, Train Accuracy: {}, Val Accuracy: {}".format(ep,
                                                                      accuracy(out[train_mask], df.data.y[train_mask]),
                                                                      accuracy(out[val_mask], df.data.y[val_mask])))
        optim.step()
        print("Train Loss:", float(loss))

def train_gin(num_layers, latent_dim):

    model = Gin(num_layers, 3703, latent_dim, 7)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fun = torch.nn.CrossEntropyLoss()

    for ep in tqdm(range(100)):
        model.train()
        optim.zero_grad()
        out = model(df.data.x, df.data.edge_index.T)
        train_mask, val_mask = df.data.train_mask, df.data.val_mask
        loss = loss_fun(out[train_mask], df.data.y[train_mask])
        loss.backward()

        print("Epoch {}, Train Accuracy: {}, Val Accuracy: {}".format(ep,
                                                                      accuracy(out[train_mask], df.data.y[train_mask]),
                                                                      accuracy(out[val_mask], df.data.y[val_mask])))
        optim.step()
        print("Train Loss:", float(loss))

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", default="gcn", help="Options: {gcn, gin}"
    )
    parser.add_argument(
        "--layers", default=3, type=int
    )
    parser.add_argument(
        "--dims", default=12, type=int, help="Latent dimension"
    )
    parser.add_argument(
        "--normalizer",
        default="row",
        type=str,
        help="Normalization method (row, col, symm)",
    )
    args = parser.parse_args()

    if args.task == "gcn":
        train_gcn(args.layers, args.dims)
    elif args.task == "gin":
        train_gin(args.layers, args.dims)