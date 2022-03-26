from baseclass import BaseGnn
import torch
import torch.nn


class Gcn(BaseGnn):
    def __init__(self, num_layers, input_dim, latent_dim, output_dim):
        super(Gcn, self).__init__(num_layers, input_dim, latent_dim, output_dim)

        self.init = torch.nn.Sequential(
            torch.nn.Linear(input_dim, latent_dim),
            torch.nn.ReLU()
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

        self.f = torch.nn.ReLU()