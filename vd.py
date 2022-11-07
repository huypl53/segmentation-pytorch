import torch
from torch.autograd import Variable
from torch import nn

class VariationalDropout(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()
        
        self.dim = dim
        self.max_alpha = alpha/(1-alpha+1e-9)
        # Initial alpha
        dummy_param = nn.Parameter(torch.empty(0))
        self.device = dummy_param.device
        
    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        
        alpha = torch.exp(self.log_alpha)
        
        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha**2 + c3 * alpha**3
        
        kl = -negative_kl
        
        return kl.mean()
    
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = Variable(torch.randn(x.size()))
            if x.is_cuda:
                epsilon = epsilon.cuda()

            dim = x.nelement()
            self.log_alpha = (torch.ones(dim).to(self.device) * self.max_alpha).log()

            # Clip alpha
            self.log_alpha = torch.clamp(self.log_alpha, max=self.max_alpha)
            alpha = torch.exp(self.log_alpha)

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x