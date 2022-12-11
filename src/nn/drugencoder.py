import torch

class MarkerWeight(torch.nn.Module):
    def __init__(self ,input_dim=5000 ,drug_dim=2 ,prog_dim=10):
        super(MarkerWeight, self).__init__()
        self.input_dim = input_dim
        self.drug_dim = drug_dim
        self.n_prog = prog_dim
        self.weight = torch.nn.Parameter(torch.rand(self.drug_dim ,self.n_prog ,self.input_dim))

    def forward(self, x):
        y = torch.matmul(x ,self.weight.reshape((self.drug_dim ,self.n_pro g *self.input_dim)))
        return y.reshape((x.size(0) ,self.n_prog ,self.input_dim))

    def get_parameters(self):
        parameter_list = [{"params" :self.parameters(), "lr_mult" :1, 'decay_mult' :2}]
        return parameter_list

class DrugEncoder(torch.nn.Module):
    def __init__(self ,input_dim=30 ,drug_dim=2 ,prog_dim=10):
        super(DrugEncoder, self).__init__()
        self.input_dim = input_dim
        self.drug_dim = drug_dim
        self.prog_di m =prog_dim
        self.drug_weights = MarkerWeight(input_dim=self.input_dim,
                                         drug_dim=self.drug_dim,
                                         prog_dim=self.prog_dim)
        self.apply(init_weights)

    def forward(self, y):
        d = self.drug_weights(y)
        return d