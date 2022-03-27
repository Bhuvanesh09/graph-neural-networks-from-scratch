from baseclass import BaseGnn
import torch


class Rnn(BaseGnn):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(Rnn, self).__init__(-1, input_dim, latent_dim, output_dim)

        self.init = torch.nn.Sequential(
            torch.nn.Linear(input_dim, latent_dim),
            torch.nn.ReLU()
        )

        self.nn1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU()
        )

        self.nn2 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU()
        )

        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, output_dim),
            torch.nn.Sigmoid(dim=1)
        )

    def initialize(self, nodes_feats):
        nodes_feats = self.nn1(nodes_feats)

        decoy_node_feats = torch.concat(
            (nodes_feats,
             torch.zeros((nodes_feats.shape[0], 1))
             ), dim=1
        )

        master_node = torch.zeros((1, self.latent_dim+1))
        master_node[0, -1] = 1

        new_graph = torch.concat(
            (master_node, decoy_node_feats),
            dim=0
        )
        return new_graph

    def aggregate(self, friends_feats, idx):
        return friends_feats[0]

    def combine(self, node_feat, message, idx):
        friend_kind = message[-1]
        self_kind = node_feat[-1]

        self_info = node_feat[:-1]

        if self_kind == 1:
            out = self.nn1(message[:-1]) + self.nn2(self_info)
        else:
            out = message[:-1]

        return torch.concat((out, self_kind), dim=0)

    def output(self, final_features):
        final_features = final_features[0, :-1]
        out = self.final_layer(final_features)
        return out

    def forward(self, nodes_feats):
        adj_list = torch.Tensor([[u, u+1] for u in range(nodes_feats.shape[0])])
        self.num_layers = nodes_feats.shape[0]

        return super(Rnn, self).forward(nodes_feats, adj_list)


