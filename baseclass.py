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

    def initialize(self, nodes_feats):
        raise NotImplementedError

    def aggregate(self, friends_feats):
        raise NotImplementedError

    def combine(self, node_feat, message):
        raise NotImplementedError

    def output(self, final_features):
        raise NotImplementedError

    def forward(self, nodes_feats, adj_list):
        latent_nodes = self.initialize(nodes_feats)

        for idx, layer in enumerate(self.layers):
            friends_combined = [[]] * nodes_feats.shape[0]

            for (fro, to) in adj_list:
                friends_combined[to].append(adj_list[fro])

            out = torch.zeros((nodes_feats.shape[0], self.latent_dim))

            for i in range(nodes_feats.shape[0]):
                friends = torch.stack(friends_combined[i])
                message = self.aggregate(friends)
                out[i] = self.combine(latent_nodes[i], message)

                latent_nodes = out

        out = self.output(latent_nodes)
        return out


