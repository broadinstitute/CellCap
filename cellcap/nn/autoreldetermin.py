##ARD regularization
import torch

class ARD_dist(torch.nn.Module):
    def __init__(self, drug_dim=2, prog_dim=10):
        super(ARD_dist, self).__init__()
        self.drug_dim = drug_dim
        self.n_prog = prog_dim
        self.weight = torch.nn.Parameter(torch.zeros(self.drug_dim, self.n_prog) + 0.5)

    def forward(self, x):
        y = torch.matmul(x, self.weight)
        return y

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list

class ARDregularizer(torch.nn.Module):
    def __init__(self, drug_dim=2, prog_dim=10):
        super(ARDregularizer, self).__init__()
        self.drug_dim = drug_dim
        self.prog_dim = prog_dim
        self.ard_dist = ARD_dist(drug_dim=self.drug_dim, prog_dim=self.prog_dim)

    def forward(self, x):
        log_alpha_ip = self.ard_dist(x)
        alpha_ip = log_alpha_ip.exp()
        return alpha_ip
