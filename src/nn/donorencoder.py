import torch

class DonorWeight(torch.nn.Module):
    def __init__(self, latent_dim=20, donor_dim=2):
        super(DonorWeight, self).__init__()
        self.latent_dim = latent_dim
        self.donor_dim = donor_dim
        self.weight = torch.nn.Parameter(torch.ones(self.donor_dim, self.latent_dim))

    def forward(self, x):
        y = torch.matmul(x, self.weight)
        return y

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list


class DonorEncoder(torch.nn.Module):
    def __init__(self, latent_dim=20, donor_dim=2):
        super(DonorEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.donor_dim = donor_dim
        self.donor_weights = DonorWeight(latent_dim=self.latent_dim,
                                         donor_dim=self.donor_dim)
        self.apply(init_weights)

    def forward(self, y):
        d = self.donor_weights(y)
        return d