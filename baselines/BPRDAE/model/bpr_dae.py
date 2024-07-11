import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRDAE(nn.Module):
    def __init__(
            self,
            n_visible1_v=4096,
            n_visible2_v=4096,
            n_hidden_v=None,
            lmb=None,
            mu=None,
            momentum=0.9,
            learning_rate=1e-3
    ) -> None:
        
        super(BPRDAE, self).__init__()

        self.momentum = momentum
        self.learning_rate = learning_rate

        self.n_visible1_v = n_visible1_v
        self.n_visible2_v = n_visible2_v
        self.n_hidden_v = n_hidden_v
        self.lamda = lmb
        self.mu = mu

        self.W1_v = nn.Parameter(torch.FloatTensor(n_visible1_v, n_hidden_v).uniform_(-4 * (6. / (n_hidden_v + n_visible1_v)) ** 0.5, 4 * (6. / (n_hidden_v + n_visible1_v)) ** 0.5))
        self.W2_v = nn.Parameter(torch.FloatTensor(n_visible2_v, n_hidden_v).uniform_(-4 * (6. / (n_hidden_v + n_visible2_v)) ** 0.5, 4 * (6. / (n_hidden_v + n_visible2_v)) ** 0.5))

        self.b1_v = nn.Parameter(torch.zeros(n_hidden_v))
        self.b2_v = nn.Parameter(torch.zeros(n_hidden_v))

        self.b1_prime_v = nn.Parameter(torch.zeros(n_visible1_v))
        self.b2_prime_v = nn.Parameter(torch.zeros(n_visible2_v))

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def get_hidden_values(
            self,
            input1_v: torch.Tensor,
            input2_v: torch.Tensor,
            input3_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        return (F.hardsigmoid(input1_v.mm(self.W1_v) + self.b1_v),
                F.hardsigmoid(input2_v.mm(self.W2_v) + self.b2_v),
                F.hardsigmoid((input3_v.mm(self.W2_v) + self.b2_v))
                )

    def get_reconstructed_input(
            self,
            hidden1_v,
            hidden2_v,
            hidden3_v
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        a = F.hardsigmoid(hidden1_v.mm(self.W1_v.T) + self.b1_prime_v)
        b = F.hardsigmoid(hidden2_v.mm(self.W2_v.T) + self.b2_prime_v)
        c = F.hardsigmoid(hidden3_v.mm(self.W2_v.T) + self.b2_prime_v)

        return a, b, c

    def get_cost_updates(
            self,
            emb_a,
            emb_b,
            emb_b1,
            backprop=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        y1_v, y2_v, y3_v, z1_v, z2_v, z3_v = self(emb_a, emb_b, emb_b1)
        y2t_v = y2_v.t()
        y3t_v = y3_v.t()

        L_x1_v = (z1_v - emb_a).pow(2).mean()
        L_x2_v = (z2_v - emb_b).pow(2).mean()
        L_x3_v = (z3_v - emb_b1).pow(2).mean()
        
        d_x1_x2_v = y1_v.mm(y2t_v).diag()
        d_x1_x3_v = y1_v.mm(y3t_v).diag()
        L_sup = F.hardsigmoid(d_x1_x2_v - d_x1_x3_v).mean()

        L_sqr = ((self.W1_v ** 2).mean() + (self.W2_v ** 2).mean() +
            + (self.b1_v ** 2).mean() + (self.b2_v ** 2).mean() +
            + (self.b1_prime_v ** 2).mean() + (self.b2_prime_v ** 2).mean()
        )
        L_123 = L_x1_v + L_x2_v + L_x3_v
        L_rec = self.mu * L_123 + self.lamda * L_sqr
        cost = L_rec - L_sup

        if backprop:
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

        return (cost,
                L_rec.item(),
                L_sup.item(),
                L_sqr.item(),
                L_123.item(),
                d_x1_x2_v.mean().item(),
                d_x1_x3_v.mean().item()
                )

    def forward(
            self, 
            emb_a: torch.Tensor, 
            emb_b: torch.Tensor, 
            emb_b1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        enc_a, enc_b, enc_b1 = self.get_hidden_values(emb_a, emb_b, emb_b1)
        rec_a, rec_b, rec_b1 = self.get_reconstructed_input(enc_a, enc_b, enc_b1)

        return enc_a, enc_b, enc_b1, rec_a, rec_b, rec_b1

    
if __name__ == '__main__':
    bprdae = BPRDAE(
        n_hidden_v=512,
        lmb=0.1,
        mu=0.1
        )
    
    tens = torch.randn(16, 3, 4096).view(-1, 4096)
    tens_exp = torch.randn(16, 3, 4096).view(-1, 4096)
    
    cost, L_rec, L_sup, L_sqr, L_123, d_x1_x2_v, d_x1_x3_v = bprdae.get_cost_updates(tens, tens, tens_exp)

    print('SUCCESS')