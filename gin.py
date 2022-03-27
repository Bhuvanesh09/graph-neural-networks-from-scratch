import torch
from baseclass import BaseGnn


class Gin(BaseGnn):
    def __init__(self, num_layers, input_dim, latent_dim, output_dim):
        super(Gin, self).__init__(num_layers, input_dim, latent_dim, output_dim)

        self.init = torch.nn.Sequential(
            torch.nn.Linear(input_dim, latent_dim),
            torch.nn.ReLU(inplace=False)
        )

        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, output_dim),
            torch.nn.Softmax(dim=1)
        )

        self.eps = torch.nn.ParameterList([])
        self.f = torch.nn.ModuleList([])

        for _ in range(self.num_layers):
            self.eps.append(torch.nn.parameter.Parameter(torch.tensor(0.5)))
            f_temp = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, latent_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(latent_dim, latent_dim),
                torch.nn.ReLU()
            )
            self.f.append(f_temp)

    def initialize(self, nodes_feats):
        return self.init(nodes_feats)

    def aggregate(self, friends_feats, idx):
        return torch.sum(friends_feats, dim=0)

    def combine(self, node_feat, message, idx):
        inter = message + (1 + self.eps[idx]) * node_feat
        out = self.f[idx](inter)
        return out

    def output(self, final_features):
        return self.final_layer(final_features)

