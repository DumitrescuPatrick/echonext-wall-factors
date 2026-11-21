"""Untrained matched-parameter prototypes; preparation requires baseline freeze."""
import json,math
from pathlib import Path
import torch
from torch import nn
from .stage2_models import ResNet1D
from .models import RFF,SPDPool,CPInteraction
from .stage2_train import sha

FAMILIES=['hilbert_rff','manifold_spd','tensor_cp','fiber_mask','fiber_coherent','transformer']

class ComparisonNet(nn.Module):
    def __init__(self,family,width):
        super().__init__()
        if family not in FAMILIES:raise ValueError('Unknown comparison')
        self.family=family;base=ResNet1D(width);target=sum(p.numel() for p in base.parameters());d=width*4
        if family=='transformer':
            embed=16 if width==8 else 24
            self.patch=nn.Conv1d(12,embed,25,25)
            layer=nn.TransformerEncoderLayer(embed,4,embed*2,dropout=0.,batch_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            position=torch.arange(100).float()[:,None];freq=torch.exp(torch.arange(0,embed,2)*(-math.log(10000)/embed));pe=torch.zeros(100,embed);pe[:,0::2]=torch.sin(position*freq);pe[:,1::2]=torch.cos(position*freq)
            self.register_buffer('position',pe);dim=embed
        else:
            self.stem=base.stem;self.blocks=base.blocks;dim=d
            if family=='hilbert_rff':self.extra=RFF(d);dim=128
            elif family=='manifold_spd':self.extra=SPDPool(d);dim=d+36
            elif family=='tensor_cp':self.extra=CPInteraction(d);dim=d+16
            else:dim=d+24
        fixed=sum(p.numel() for p in self.parameters());hidden=round((target-fixed-1)/(dim+2))
        if hidden<4:raise ValueError('Parameter target cannot support comparison head')
        self.head=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Linear(hidden,1))
        if abs(sum(p.numel() for p in self.parameters())-target)/target>.005:raise ValueError('Parameter budget mismatch')
    def forward(self,x,query=None,available=None):
        if x.ndim!=3 or x.shape[1:]!=(12,2500):raise ValueError('Expected N x 12 x 2500')
        q=torch.ones(len(x),12,device=x.device,dtype=x.dtype) if query is None else query
        a=torch.ones_like(q) if available is None else available
        if q.shape!=(len(x),12) or a.shape!=q.shape:raise ValueError('Mask shape mismatch')
        if not torch.isfinite(x).all() or not torch.all((q==0)|(q==1)) or not torch.all((a==0)|(a==1)):raise ValueError('Nonfinite input or invalid lead mask')
        x=x*(q*a).unsqueeze(-1)
        if self.family=='transformer':
            z=self.patch(x).transpose(1,2);features=self.encoder(z+self.position).mean(1)
        else:
            z=self.blocks(self.stem(x));pool=z.mean(-1)
            if self.family=='hilbert_rff':features=self.extra(pool)
            elif self.family in ('manifold_spd','tensor_cp'):features=torch.cat([pool,self.extra(z)],1)
            else:features=torch.cat([pool,q,q*a],1)
        return self.head(features).squeeze(-1)

def prepare():
    freeze=Path('reports/stage2_v1/BASELINE_FROZEN.json')
    if not freeze.exists():raise PermissionError('Freeze and verify the completed ResNet baseline first')
    f=json.loads(freeze.read_text());out=Path('reports/stage2_v1/geometric_comparison_plan.json')
    if out.exists():raise FileExistsError('Comparison plan already frozen')
    caps=[]
    for i in range(6):
        r=json.loads(Path(f'reports/stage2_v1/tune_{i}_complete.json').read_text())
        caps.append({'candidate_index':i,'width':r['configuration']['width'],'completed_epochs':r['epochs_completed'],'completed_epoch_cpu_seconds':sum(h['cpu_seconds'] for h in json.loads(Path(f'reports/stage2_v1/tune_{i}_history.json').read_text())),'maximum_training_epochs':20})
    finals=[]
    for seed in f['final_seeds']:
        r=json.loads(Path(f'reports/stage2_v1/seed_{seed}_complete.json').read_text());finals.append({'seed':seed,'baseline_cpu_seconds':r['total_cpu_seconds']})
    plan={'status':'PREPARED_NOT_TRAINED','baseline_freeze_sha256':sha(freeze),'families':FAMILIES,'train':'same 72475 official train; patient weights; no_split excluded','validation':'same 4626 official validation; same selection/stopping/calibration roles','test':'sealed; unavailable','trials_per_family':6,'final_seeds':f['final_seeds'],'capacity_tiers':[8,16],'parameter_matching':'within 0.5% of corresponding baseline width; count buffers/operations separately','candidate_reference_caps':caps,'final_seed_reference_caps':finals,'strict_compute_rule':'Each corresponding candidate/final seed receives 1.05 times reference completed-process CPU budget, on one CPU thread; count density fitting and coherence/audit forwards against it; stop at first epoch/compute cap and keep last complete checkpoint. Validate/report overshoot per indivisible step. Audit postprocessing and data-cache creation costs reported separately, never free.','important_limit':'Reference tuning CPU excludes discarded interrupted partial-epoch work and cache creation; full operational tuning expense is a lower bound, not an exact measured matched-compute experiment. Future comparisons must log all work and distinguish useful optimization from discarded overhead.','additional_exposure_control':'Report both strict-compute and matched-epoch/input-exposure analyses separately if run; different architectures/costs cannot simultaneously match every resource','primary_comparisons':'Hilbert/SPD/tensor/fiber-mask versus frozen ResNet; Transformer architecture comparator; coherent versus mask is a separately labelled mechanism contrast','conditional_fiber_status':'mask-aware model implemented; clinical completion estimator/coherence trainer still needs training-only patient-excluded implementation and tests; no superiority claim','source':'stage2_geometric.py; synthetic correctness only'}
    import time
    torch.set_num_threads(1);bench=[]
    for width in [8,16]:
        for family in ['resnet']+FAMILIES:
            torch.manual_seed(57001)
            model=ResNet1D(width) if family=='resnet' else ComparisonNet(family,width)
            opt=torch.optim.AdamW(model.parameters(),lr=.001);x=torch.randn(64,12,2500);y=(torch.arange(64)%2).float()
            def step():
                opt.zero_grad();torch.nn.functional.binary_cross_entropy_with_logits(model(x),y).backward();opt.step()
            step();t=time.process_time()
            for _ in range(3):step()
            bench.append({'width':width,'family':family,'synthetic_batch':64,'supervised_step_cpu_seconds':(time.process_time()-t)/3,'trainable_parameters':sum(p.numel() for p in model.parameters()),'fixed_buffer_elements':sum(p.numel() for p in model.buffers()),'clinical_data_used':False,'conditional_completion_cost_included':False})
    plan['synthetic_step_benchmark']=bench
    out.write_text(json.dumps(plan,indent=2)+'\n')

if __name__=='__main__':prepare()
