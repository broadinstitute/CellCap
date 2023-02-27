import torch

class MarkerWeight(torch.nn.Module):
    def __init__(self, latent_dim=20, drug_dim=2, prog_dim=10, key=True):
        super(MarkerWeight, self).__init__()
        self.latent_dim = latent_dim
        self.drug_dim = drug_dim
        self.n_prog = prog_dim
        if key:
            self.weight = torch.nn.Parameter(torch.rand(self.drug_dim, self.n_prog, self.latent_dim))
        else:
            self.weight = torch.nn.Parameter(torch.zeros(self.drug_dim, self.n_prog, self.latent_dim))

    def forward(self, x):
        y = torch.matmul(x, self.weight.reshape((self.drug_dim, self.n_prog * self.latent_dim)))
        return y.reshape((x.size(0), self.n_prog, self.latent_dim))

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list


class DrugEncoder(torch.nn.Module):
    def __init__(self, latent_dim=20, drug_dim=2, prog_dim=10, key=True):
        super(DrugEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.drug_dim = drug_dim
        self.prog_dim = prog_dim
        self.drug_weights = MarkerWeight(latent_dim=self.latent_dim,
                                         drug_dim=self.drug_dim,
                                         prog_dim=self.prog_dim, key=key)

    def forward(self, y):
        d = self.drug_weights(y)
        return d
