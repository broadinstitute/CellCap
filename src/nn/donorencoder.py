import torch

class DonorWeight(torch.nn.Module):
    def __init__(self, input_dim=5000, drug_dim=2):
        super(DonorWeight, self).__init__()
        self.input_dim = input_dim
        self.drug_dim = drug_dim
        self.weight = torch.nn.Parameter(torch.ones(self.drug_dim, self.input_dim))

    def forward(self, x):
        y = torch.matmul(x, self.weight)
        return y

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list


class DonorEncoder(torch.nn.Module):
    def __init__(self, input_dim=30, drug_dim=2):
        super(DonorEncoder, self).__init__()
        self.input_dim = input_dim
        self.drug_dim = drug_dim
        self.drug_weights = DonorWeight(input_dim=self.input_dim,
                                        drug_dim=self.drug_dim)
        self.apply(init_weights)

    def forward(self, y):
        d = self.drug_weights(y)
        return d