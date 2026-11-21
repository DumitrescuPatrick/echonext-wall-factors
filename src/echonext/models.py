"""Synthetic-tested research prototypes, not trained or validated clinical models."""
import math
import torch
from torch import nn
from torch.nn import functional as F

FAMILIES=('cnn','resnet','transformer','hilbert_rff','manifold_spd','tensor_cp','fiber_mask','fiber_coherent')


def seed_everything(seed):
    torch.manual_seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


class Stem(nn.Module):
    def __init__(self,width=24):
        super().__init__();self.output=width*2
        self.net=nn.Sequential(nn.Conv1d(12,width,15,5,7),nn.GroupNorm(4,width),nn.ReLU(),nn.Conv1d(width,width*2,7,2,3),nn.GroupNorm(4,width*2),nn.ReLU(),nn.Conv1d(width*2,width*2,5,2,2),nn.ReLU())
    def forward(self,x):return self.net(x)


class Residual(nn.Module):
    def __init__(self,d):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(d,d,5,padding=2),nn.GroupNorm(4,d),nn.ReLU(),nn.Conv1d(d,d,5,padding=2),nn.GroupNorm(4,d))
    def forward(self,x):return F.relu(x+self.net(x))


class RFF(nn.Module):
    """Finite Monte Carlo features for a Gaussian RKHS; not an analytic-signal transform."""
    def __init__(self,d,k=128,seed=61001):
        super().__init__();g=torch.Generator().manual_seed(seed)
        self.register_buffer('omega',torch.randn(d,k,generator=g)/math.sqrt(d))
        self.register_buffer('phase',2*math.pi*torch.rand(k,generator=g));self.k=k
    def forward(self,x):return math.sqrt(2/self.k)*torch.cos(x@self.omega+self.phase)


class SymmetricLog(torch.autograd.Function):
    """SPD matrix log with Frechet derivative stable at repeated eigenvalues."""
    @staticmethod
    def forward(ctx,x):
        values,vectors=torch.linalg.eigh((x+x.transpose(-1,-2))/2)
        if torch.any(values<=0):raise ValueError('Matrix logarithm requires SPD input')
        ctx.save_for_backward(values,vectors)
        return (vectors*values.log().unsqueeze(-2))@vectors.transpose(-1,-2)
    @staticmethod
    def backward(ctx,gradient):
        v,u=ctx.saved_tensors;vi=v.unsqueeze(-1);vj=v.unsqueeze(-2);delta=vi-vj
        near=delta.abs()<=1e-5*torch.maximum(vi,vj)
        denom=torch.where(near,torch.ones_like(delta),delta)
        divided=torch.where(near,2/(vi+vj),(vi.log()-vj.log())/denom)
        g=(gradient+gradient.transpose(-1,-2))/2
        return u@(divided*(u.transpose(-1,-2)@g@u))@u.transpose(-1,-2)


class SPDPool(nn.Module):
    def __init__(self,d,k=8):
        super().__init__();self.project=nn.Conv1d(d,k,1);self.k=k
        ij=torch.triu_indices(k,k);self.register_buffer('ii',ij[0]);self.register_buffer('jj',ij[1])
        self.register_buffer('scale',torch.where(ij[0]==ij[1],1.,math.sqrt(2)))
    def forward(self,x):
        z=self.project(x);z=z-z.mean(-1,keepdim=True)
        cov=z@z.transpose(1,2)/max(1,z.shape[-1]-1)
        trace=cov.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(1e-6)
        cov=cov/trace[:,None,None]*self.k
        eye=torch.eye(self.k,device=x.device,dtype=x.dtype)
        log=SymmetricLog.apply(.9*cov+.1*eye)
        return log[:,self.ii,self.jj]*self.scale


class CPInteraction(nn.Module):
    def __init__(self,d,rank=16):
        super().__init__();self.parts=nn.ModuleList(nn.Linear(d,rank) for _ in range(3));self.rank=rank
    def forward(self,x):
        sections=torch.tensor_split(x,3,dim=-1)
        return torch.stack([torch.tanh(p(z.mean(-1))) for p,z in zip(self.parts,sections)]).prod(0)/math.sqrt(self.rank)


class Features(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='transformer':
            self.patch=nn.Conv1d(12,48,25,25)
            layer=nn.TransformerEncoderLayer(48,4,96,dropout=0.,batch_first=True,norm_first=False)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            position=torch.arange(100).float().unsqueeze(1);frequency=torch.exp(torch.arange(0,48,2)*(-math.log(10000)/48))
            pe=torch.zeros(100,48);pe[:,0::2]=torch.sin(position*frequency);pe[:,1::2]=torch.cos(position*frequency)
            self.register_buffer('position',pe);self.dim=48;return
        self.stem=Stem();self.dim=48
        if kind=='resnet':self.extra=nn.Sequential(Residual(48),Residual(48))
        elif kind=='hilbert_rff':self.extra=RFF(48);self.dim=128
        elif kind=='manifold_spd':self.extra=SPDPool(48);self.dim=84
        elif kind=='tensor_cp':self.extra=CPInteraction(48);self.dim=64
        elif kind in ('fiber_mask','fiber_coherent'):self.dim=72
    def forward(self,x,query,available):
        if self.kind=='transformer':
            z=self.patch(x).transpose(1,2);return self.encoder(z+self.position[:z.shape[1]]).mean(1)
        z=self.stem(x)
        if self.kind=='resnet':z=self.extra(z)
        pooled=z.mean(-1)
        if self.kind=='hilbert_rff':return self.extra(pooled)
        if self.kind in ('manifold_spd','tensor_cp'):return torch.cat([pooled,self.extra(z)],dim=1)
        if self.kind in ('fiber_mask','fiber_coherent'):return torch.cat([pooled,query,query*available],dim=1)
        return pooled


class ECGModel(nn.Module):
    def __init__(self,kind,target_parameters=100000,outputs=1):
        super().__init__()
        if kind not in FAMILIES:raise ValueError('Unknown model family')
        self.kind=kind;self.features=Features(kind)
        base=sum(p.numel() for p in self.features.parameters());width=(target_parameters-base-outputs)//(self.features.dim+1+outputs)
        if width<8:raise ValueError('Parameter target too small')
        self.head=nn.Sequential(nn.Linear(self.features.dim,width),nn.ReLU(),nn.Linear(width,outputs))
    def forward(self,x,query=None,available=None):
        if x.ndim!=3 or x.shape[1:]!=(12,2500):raise ValueError('Expected batch x 12 x 2500')
        b=x.shape[0];query=torch.ones(b,12,device=x.device,dtype=x.dtype) if query is None else query
        available=torch.ones_like(query) if available is None else available
        if query.shape!=(b,12) or available.shape!=(b,12):raise ValueError('Invalid lead mask')
        if not torch.all((query==0)|(query==1)) or not torch.all((available==0)|(available==1)):raise ValueError('Lead masks must be binary')
        if not torch.isfinite(x).all():raise ValueError('Nonfinite input')
        visible=x*(query*available).unsqueeze(-1)
        return self.head(self.features(visible,query,available))


def masked_bce(logits,targets,weights=None):
    if logits.shape!=targets.shape:raise ValueError('Outcome shape mismatch')
    known=torch.isfinite(targets)
    if not known.any():raise ValueError('No observed targets')
    if torch.any(known & (targets!=0) & (targets!=1)):raise ValueError('Observed targets must be binary')
    safe=torch.where(known,targets,torch.zeros_like(targets))
    loss=F.binary_cross_entropy_with_logits(logits,safe,reduction='none')
    w=torch.ones_like(loss) if weights is None else torch.broadcast_to(weights,loss.shape)
    if torch.any(w<0) or not torch.isfinite(w).all() or (w*known).sum()<=0:raise ValueError('Invalid loss weights')
    return (loss*w*known).sum()/(w*known).sum()


def tower_penalty(parent,children,weights):
    """Squared finite-completion residual; NOT an unbiased exact tower error."""
    if children.ndim!=3 or parent.shape!=(children.shape[0],children.shape[2]) or weights.shape!=children.shape[:2]:raise ValueError('Constraint shape mismatch')
    if not torch.isfinite(parent).all() or not torch.isfinite(children).all() or torch.any((parent<0)|(parent>1)) or torch.any((children<0)|(children>1)):raise ValueError('Tower residual requires finite probabilities')
    if torch.any(weights<0) or not torch.allclose(weights.sum(1),torch.ones_like(weights[:,0])):raise ValueError('Invalid conditional weights')
    return ((parent-(children*weights.unsqueeze(-1)).sum(1))**2).mean()
