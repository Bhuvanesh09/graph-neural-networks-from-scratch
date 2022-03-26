import torch_geometric as pyg
import pytorch_lightning as pl
from gcn import Gcn
from torch.utils.data import Dataset, DataLoader

class GraphData(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.df

df = pyg.datasets.Planetoid("datasets/planetoid", "citeseer")

model = Gcn(2, 3703, 20, 7)

trainer = pl.Trainer()
data = df.data
# data = (data.x, data.edge_list, )
data = GraphData(data)
loader = DataLoader(data, batch_size=1, collate_fn=lambda x: x)

trainer.fit(model, loader)