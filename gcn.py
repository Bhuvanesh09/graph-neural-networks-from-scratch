from baseclass import BaseGnn
import torch
import torch.nn


class Gcn(BaseGnn):
    def __init__(self, num_layers, input_dim, latent_dim, output_dim):
        super(Gcn, self).__init__(num_layers, input_dim, latent_dim, output_dim)

        self.init = torch.nn.Sequential(
            torch.nn.Linear(input_dim, latent_dim),
            torch.nn.ReLU(inplace=False)
        )

        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, output_dim),
            torch.nn.Softmax()
        )

        self.layers = torch.nn.ModuleList([])

        for i in range(self.num_layers):
            w = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
            b = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
            self.layers.append(
                torch.nn.ModuleDict({
                    "w": w,
                    "b": b,
                })
            )

        self.f = torch.nn.ReLU(inplace=False)

    def initialize(self, nodes_feats):
        return self.init(nodes_feats)

    def aggregate(self, friends_feats, idx):
        # GCN has mean of the features as its aggregation
        return torch.mean(friends_feats, dim=0)

    def combine(self, node_feat, message, idx):
        w, b = self.layers[idx]["w"], self.layers[idx]["b"]
        return self.f(
            w(message) + b(node_feat)
        )

    def output(self, final_features):
        # Unlike chemistry applications pooling is not reqd since we are
        # doing node level classification
        return self.final_layer(final_features)
