"""Stage 2 standard temporal ResNet; Stage 1 model definitions stay frozen."""
import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self,a,b,kernel,stride):
        super().__init__()
        self.main=nn.Sequential(nn.Conv1d(a,b,kernel,stride,kernel//2,bias=False),nn.GroupNorm(4,b),nn.ReLU(),nn.Conv1d(b,b,kernel,1,kernel//2,bias=False),nn.GroupNorm(4,b))
        self.skip=nn.Identity() if a==b and stride==1 else nn.Conv1d(a,b,1,stride,bias=False)
    def forward(self,x):return F.relu(self.main(x)+self.skip(x))

class ResNet1D(nn.Module):
    def __init__(self,width=16):
        super().__init__()
        if width not in (8,16):raise ValueError('Unplanned ResNet width')
        self.stem=nn.Sequential(nn.Conv1d(12,width,31,10,15,bias=False),nn.GroupNorm(4,width),nn.ReLU())
        self.blocks=nn.Sequential(BasicBlock(width,width*2,3,2),BasicBlock(width*2,width*4,3,2),BasicBlock(width*4,width*4,3,1))
        self.head=nn.Sequential(nn.Linear(width*4,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(12,2500):raise ValueError('Expected N x 12 x 2500')
        return self.head(self.blocks(self.stem(x)).mean(-1)).squeeze(-1)
