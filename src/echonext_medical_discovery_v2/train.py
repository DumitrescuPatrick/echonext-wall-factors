"""MPS-only clinical unsupervised runner; no silent fallback or test-set access."""
import argparse,json,random,hashlib,platform
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from echonext_unsupervised_discovery_v1.data import Waves,roles,sha
from .models import CompactVAE,CompactSimCLR,input_view,augment,contrastive
from .latent import CACHE,CONFIG

def device():
    import os
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise RuntimeError('Silent CPU fallback prohibited')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable in this process; run from native Apple Silicon Terminal. Clinical CPU substitution is prohibited.')
    return torch.device('mps')

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['vae','simclr'],required=True);p.add_argument('--view',choices=['native','per_record_lead_unit_SD'],required=True);p.add_argument('--seed',type=int,choices=[69001,69002,69003,69004,69005],required=True);a=p.parse_args();dev=device();root=Path('private/medical_discovery_next_phase_unsupervised')/f'{a.method}_{a.view}_{a.seed}';root.mkdir(parents=True,exist_ok=True,mode=0o700)
    if (root/'COMPLETE.json').exists():raise FileExistsError('Completed run is immutable')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.mps.manual_seed(a.seed);torch.use_deterministic_algorithms(True)
    integrity=json.loads(Path('reports/stage2_v1/development_cache_integrity.json').read_text())
    for name,expected in integrity.items():
        if sha(CACHE/name)!=expected['sha256']:raise ValueError('Training cache/source mismatch')
    data=Waves(CACHE,'train');fit,monitor,_=roles(data.patients);_,first=np.unique(np.array(data.patients)[monitor],return_index=True);monitor=monitor[np.sort(first)];train=Waves(CACHE,'train',fit);stop=Waves(CACHE,'train',monitor);g=train.groups[train.indices];unique,counts=np.unique(g,return_counts=True);weights=dict(zip(unique,1/counts));sampler=WeightedRandomSampler([weights[v] for v in g],len(train),replacement=True);loader=DataLoader(train,batch_size=64,sampler=sampler,num_workers=0,pin_memory=False);monitor_loader=DataLoader(stop,batch_size=64,num_workers=0,pin_memory=False)
    model=(CompactVAE() if a.method=='vae' else CompactSimCLR()).to(dev);optimizer=torch.optim.Adam(model.parameters(),lr=.001);history=[];best=float('inf');bad=0;start=0
    source_files=[Path(__file__),Path(__file__).with_name('models.py'),CONFIG,Path(__file__).parents[1]/'echonext_unsupervised_discovery_v1/data.py',Path(__file__).parents[1]/'echonext/data.py'];source_hash={str(f):sha(f) for f in source_files};software={'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()};last=root/'last.pt'
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state['source_hash']!=source_hash or state['device']!='mps' or state.get('software')!=software:raise ValueError('Resume source/backend mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for slot in optimizer.state.values():
            for k,v in slot.items():
                if torch.is_tensor(v):slot[k]=v.cpu() if k=='step' and not optimizer.defaults.get('capturable',False) else v.to(dev)
        history=state['history'];best=state['best'];bad=state['bad'];start=30 if bad>=5 else state['epoch'];torch.set_rng_state(state['CPU_rng']);torch.mps.set_rng_state(state['MPS_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    def objective(x,group,step):
        if a.method=='vae':return model.objective(x)
        return contrastive(model(augment(x,(step*7)%51-25)),model(augment(x,(step*17+13)%51-25)),group)
    for epoch in range(start,30):
        model.train();total=torch.zeros((),device=dev);n=0
        for step,(x,group) in enumerate(loader):
            x=input_view(x.to(dev),a.view);group=group.to(dev);optimizer.zero_grad();loss=objective(x,group,step+epoch*100003)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
            loss.backward();optimizer.step();total+=loss.detach()*len(x);n+=len(x)
        saved_cpu=torch.get_rng_state();saved_mps=torch.mps.get_rng_state();torch.manual_seed(a.seed+999);torch.mps.manual_seed(a.seed+999);model.eval();val=torch.zeros((),device=dev);vn=0
        with torch.no_grad():
            for step,(x,group) in enumerate(monitor_loader):
                x=input_view(x.to(dev),a.view);group=group.to(dev);value=objective(x,group,step);val+=value*len(x);vn+=len(x)
        torch.set_rng_state(saved_cpu);torch.mps.set_rng_state(saved_mps);v=float((val/vn).cpu());history.append({'epoch':epoch+1,'training_loss':float((total/n).cpu()),'monitor_loss':v})
        if v<best-1e-5:
            best=v;bad=0;torch.save({'model':{k:t.detach().cpu().clone() for k,t in model.state_dict().items()},'method':a.method,'view':a.view,'seed':a.seed,'source_hash':source_hash,'software':software},root/'best.pt')
        else:bad+=1
        state={'model':{k:t.detach().cpu().clone() for k,t in model.state_dict().items()},'optimizer':optimizer.state_dict(),'epoch':epoch+1,'history':history,'best':best,'bad':bad,'source_hash':source_hash,'device':'mps','software':software,'CPU_rng':torch.get_rng_state(),'MPS_rng':torch.mps.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()};torch.save(state,root/'last.tmp.pt');(root/'last.tmp.pt').replace(last);(root/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(history[-1]),flush=True)
        if bad>=5:break
    (root/'COMPLETE.json').write_text(json.dumps({'method':a.method,'view':a.view,'seed':a.seed,'epochs':len(history),'best_monitor_loss':best,'labels_used':False,'validation_or_test_used_for_training':False,'source_hash':source_hash,'software':software},indent=2))
if __name__=='__main__':main()
