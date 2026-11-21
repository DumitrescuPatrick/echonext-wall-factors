"""Compact matched encoders; unsupervised objectives never receive echo labels."""
import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(12,16,15,5,7,bias=False),nn.GroupNorm(4,16),nn.ReLU(),nn.Conv1d(16,32,9,5,4,bias=False),nn.GroupNorm(4,32),nn.ReLU(),nn.Conv1d(32,48,5,2,2,bias=False),nn.GroupNorm(4,48),nn.ReLU(),nn.AdaptiveAvgPool1d(8),nn.Flatten())
    def forward(self,x):return self.net(x)

class CompactVAE(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder();self.mu=nn.Linear(384,32);self.logvar=nn.Linear(384,32);self.decoder=nn.Sequential(nn.Linear(32,48*50),nn.Unflatten(1,(48,50)),nn.Upsample(size=500,mode='linear',align_corners=False),nn.Conv1d(48,24,9,padding=4),nn.ReLU(),nn.Upsample(size=2500,mode='linear',align_corners=False),nn.Conv1d(24,12,15,padding=7))
    def embedding(self,x):return self.mu(self.encoder(x))
    def forward(self,x):
        h=self.encoder(x);mu=self.mu(h);lv=self.logvar(h).clamp(-12,12);z=mu+torch.randn_like(mu)*torch.exp(.5*lv) if self.training else mu;return self.decoder(z),mu,lv
    def objective(self,x):
        reconstruction,mu,lv=self(x);return (reconstruction-x).square().mean()+.001*(-.5*(1+lv-mu.square()-lv.exp())).mean()

class CompactSimCLR(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder();self.latent=nn.Linear(384,32);self.projector=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,16))
    def embedding(self,x):return self.latent(self.encoder(x))
    def forward(self,x):return F.normalize(self.projector(self.embedding(x)),dim=1)


def input_view(x,view):
    if view=='native':return x
    if view=='per_record_lead_unit_SD':
        z=x-x.mean(-1,keepdim=True);return z/z.std(-1,unbiased=False,keepdim=True).clamp_min(1e-4)
    raise ValueError('Unknown frozen input view')


def augment(x,shift):
    # Synchronized reflect-padded translation; no circular wrapping or polarity swaps.
    n=x.shape[-1];padded=F.pad(x,(25,25),mode='reflect');shifted=padded[:,:,25+shift:25+shift+n];gain=.9+.2*torch.rand((len(x),1,1),device=x.device);noise=torch.randn_like(x)*(.01*x.std(-1,unbiased=False,keepdim=True));return shifted*gain+noise


def contrastive(a,b,patients):
    n=len(a);z=torch.cat([a,b]);target=(torch.arange(2*n,device=z.device)+n)%(2*n);groups=torch.cat([patients,patients]);allowed=groups[:,None]!=groups[None,:];allowed[torch.arange(2*n,device=z.device),target]=True;return F.cross_entropy((z@z.T/.2).masked_fill(~allowed,-torch.inf),target)
