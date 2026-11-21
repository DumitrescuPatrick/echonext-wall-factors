"""Synthetic float32 ResNet throughput; no patient/test data or silent fallback."""
import argparse,json,time
from pathlib import Path
import torch
from echonext.stage2_models import ResNet1D
from .pipeline import device,seed_all

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
    torch.set_num_threads(4)
    for name in ('cpu','mps'):
        try:dev=device(name)
        except RuntimeError as e:records.append({'device':name,'available':False,'reason':str(e)});continue
        for batch in (16,32,64):
            seed_all(62001);model=ResNet1D(16).to(dev);optimizer=torch.optim.Adam(model.parameters(),lr=.001);x=torch.randn(batch,12,2500,device=dev);y=torch.zeros(batch,device=dev)
            for step in range(8):
                if step==3:
                    if name=='mps':torch.mps.synchronize()
                    start=time.perf_counter()
                optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y);loss.backward();optimizer.step()
            if name=='mps':torch.mps.synchronize()
            elapsed=time.perf_counter()-start;records.append({'device':name,'available':True,'batch':batch,'examples_per_second':5*batch/elapsed,'steps_measured':5,'warmup_steps':3,'synthetic':True})
    a.output.write_text(json.dumps({'records':records,'caveat':'Synthetic single-process compute only; data-loader throughput and full-model numerical equivalence not measured'},indent=2));print('Synthetic benchmark report saved; no clinical training performed.')
if __name__=='__main__':main()
