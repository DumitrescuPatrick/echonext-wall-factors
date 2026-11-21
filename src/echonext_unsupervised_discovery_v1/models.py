"""Float32 waveform encoders and label-free objectives."""
import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(12,32,15,5,7,bias=False),nn.GroupNorm(4,32),nn.ReLU(),
            nn.Conv1d(32,64,9,5,4,bias=False),nn.GroupNorm(4,64),nn.ReLU(),
            nn.Conv1d(64,64,5,2,2,bias=False),nn.GroupNorm(4,64),nn.ReLU(),nn.AdaptiveAvgPool1d(8),nn.Flatten())
        self.output = nn.Linear(512,latent,bias=False)
    def forward(self,x): return self.output(self.net(x))

class Autoencoder(nn.Module):
    def __init__(self, latent=32, variational=False):
        super().__init__(); self.variational=variational; self.encoder=Encoder(latent)
        self.logvar=nn.Linear(latent,latent) if variational else None
        self.decoder=nn.Sequential(nn.Linear(latent,64*50),nn.Unflatten(1,(64,50)),
            nn.Upsample(size=500,mode='linear',align_corners=False),nn.Conv1d(64,32,9,padding=4),nn.ReLU(),
            nn.Upsample(size=2500,mode='linear',align_corners=False),nn.Conv1d(32,12,15,padding=7))
    def forward(self,x):
        mu=self.encoder(x); lv=self.logvar(mu).clamp(-12,12) if self.variational else torch.zeros_like(mu)
        z=mu+torch.randn_like(mu)*torch.exp(.5*lv) if self.training and self.variational else mu
        return self.decoder(z),mu,lv
    def loss(self,x,beta=.001):
        reconstruction,mu,lv=self(x)
        return (reconstruction-x).square().mean()+beta*(-.5*(1+lv-mu.square()-lv.exp())).mean()

class IDEC(nn.Module):
    def __init__(self,k=4,latent=32):
        super().__init__(); self.ae=Autoencoder(latent); self.centers=nn.Parameter(torch.zeros(k,latent))
    def forward(self,x):
        reconstruction,z,_=self.ae(x); q=1/(1+(z[:,None]-self.centers).square().sum(-1)); q=q/q.sum(1,keepdim=True)
        return reconstruction,z,q
    @staticmethod
    def target(q):
        p=q.square()/q.sum(0,keepdim=True).clamp_min(1e-8); return (p/p.sum(1,keepdim=True)).detach()
    def loss(self,x,target,gamma=.1):
        r,_,q=self(x); return (r-x).square().mean()+gamma*F.kl_div(q.clamp_min(1e-8).log(),target,reduction='batchmean')

class SimCLR(nn.Module):
    def __init__(self,latent=32):
        super().__init__(); self.encoder=Encoder(latent); self.projector=nn.Sequential(nn.Linear(latent,64),nn.ReLU(),nn.Linear(64,32))
    def forward(self,x): return F.normalize(self.projector(self.encoder(x)),dim=1)

def augment(x):
    # Shared temporal shift across leads; no lead permutation or polarity flips.
    gain=1+.05*torch.randn((len(x),1,1),device=x.device)
    return torch.roll(x,int(torch.randint(-12,13,()).item()),dims=-1)*gain+.01*torch.randn_like(x)

def contrastive_loss(a,b,patients,temperature=.2):
    z=torch.cat((a,b)); n=len(a); sim=z@z.T/temperature
    group=torch.cat((patients,patients)); positive=(torch.arange(2*n,device=z.device)+n)%(2*n)
    allowed=group[:,None]!=group[None,:]; allowed[torch.arange(2*n,device=z.device),positive]=True
    return F.cross_entropy(sim.masked_fill(~allowed,-torch.inf),positive)

class DeepSVDD(nn.Module):
    def __init__(self,latent=32):
        super().__init__(); self.encoder=Encoder(latent)
        # No biases or affine normalization: reduce constant-output shortcuts.
        for m in self.encoder.modules():
            if isinstance(m,nn.GroupNorm): m.register_parameter('weight',None); m.register_parameter('bias',None); m.affine=False
        self.register_buffer('center',torch.zeros(latent))
    def forward(self,x): return (self.encoder(x)-self.center).square().sum(1)
