import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn


class BaseGnn(pl.LightningModule):
    def __init__(self, num_layers, input_dim, latent_dim, output_dim):
        super().__init__()
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def initialize(self, nodes_feats):
        raise NotImplementedError

    def aggregate(self, friends_feats, idx):
        raise NotImplementedError

    def combine(self, node_feat, message, idx):
        raise NotImplementedError

    def output(self, final_features):
        raise NotImplementedError

    def forward(self, nodes_feats, adj_list):
        latent_nodes = self.initialize(nodes_feats)

        for idx, layer in enumerate(self.layers):
            friends_combined = [[]] * nodes_feats.shape[0]

            for (fro, to) in adj_list:
                friends_combined[to].append(latent_nodes[fro])

            out = torch.zeros((nodes_feats.shape[0], self.latent_dim))

            for i in range(nodes_feats.shape[0]):
                friends = torch.stack(friends_combined[i])
                message = self.aggregate(friends, idx)
                out[i] = self.combine(latent_nodes[i], message, idx)

                latent_nodes = out

        out = self.output(latent_nodes)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


    def training_step(self, data):
        data = data[0]
        node_features = data.x
        adj_list = data.edge_index.T
        y = data.y
        out = self.forward(node_features, adj_list)

        train_out = out[data.train_mask]
        train_y = y[data.train_mask]

        loss = self.loss_fn(train_out, train_y)
        self.log("train_loss", loss)

        val_out = out[data.val_mask]
        val_y = y[data.val_mask]

        loss2 = self.loss_fn(val_out, val_y)

        self.log("val_loss", loss2)
        return loss




