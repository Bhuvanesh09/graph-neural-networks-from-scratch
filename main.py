import torch_geometric as pyg
import torch
import pytorch_lightning as pl
from gcn import Gcn
from torch.utils.data import Dataset, DataLoader


class GraphData(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return (self.df.x, self.df.edge_index, self.df.y, self.df.train_mask, self.df.val_mask)


df = pyg.datasets.Planetoid("datasets/planetoid", "citeseer")

model = Gcn(2, 3703, 7, 7)

trainer = pl.Trainer(gpus=1)
data = df.data
# data = (data.x, data.edge_list, )
data = GraphData(data)
loader = DataLoader(data, batch_size=1, collate_fn=lambda x: x)
torch.autograd.set_detect_anomaly(True)
trainer.fit(model, loader)