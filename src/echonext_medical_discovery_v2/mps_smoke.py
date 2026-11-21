"""Synthetic device/checkpoint smoke; no clinical waveforms are loaded."""
import tempfile,json
from pathlib import Path
import torch
from .models import CompactVAE,CompactSimCLR,contrastive,augment
from .train import device

def synthetic_smoke(dev):
    results=[]
    for name,model in [('vae',CompactVAE()),('simclr',CompactSimCLR())]:
        model=model.to(dev);optimizer=torch.optim.Adam(model.parameters(),lr=.001);x=torch.randn(4,12,2500,device=dev);groups=torch.arange(4,device=dev);optimizer.zero_grad();loss=model.objective(x) if name=='vae' else contrastive(model(augment(x,3)),model(augment(x,-8)),groups);loss.backward();optimizer.step()
        if loss.device.type!=dev.type or any(p.device.type!=dev.type for p in model.parameters()):raise RuntimeError('Unexpected device transfer')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint.pt';torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict()},path);loaded=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer'])
            # A second step exercises optimizer-state restoration on this backend.
            optimizer.zero_grad();z=model.objective(x) if name=='vae' else contrastive(model(x),model(x+.001),groups);z.backward();optimizer.step()
        results.append({'method':name,'device':str(dev),'finite':bool(torch.isfinite(z).detach().cpu()),'checkpoint_resume_step_passed':True})
    return results

def main():
    dev=device();print(json.dumps({'synthetic_only':True,'results':synthetic_smoke(dev)},indent=2))
if __name__=='__main__':main()
