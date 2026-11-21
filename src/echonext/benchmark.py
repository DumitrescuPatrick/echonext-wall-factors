"""Artificial-array CPU resource benchmark, never an EchoNext training experiment."""
import argparse,json,resource,subprocess,sys,time
from pathlib import Path
import torch
from .models import ECGModel,FAMILIES,seed_everything,masked_bce

def one(family):
    seed_everything(51001);m=ECGModel(family);x=torch.randn(16,12,2500);y=(torch.arange(16)%2).float()[:,None]
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
    def step():
        optimizer.zero_grad();loss=masked_bce(m(x),y);loss.backward();optimizer.step()
    step();t=time.perf_counter()
    for _ in range(3):step()
    seconds=(time.perf_counter()-t)/3
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin':peak*=1024
    return {'family':family,'synthetic_batch':16,'trainable_parameters':sum(p.numel() for p in m.parameters()),'fixed_buffer_elements':sum(p.numel() for p in m.buffers()),'step_seconds':seconds,'peak_process_rss_bytes':peak,'device':'cpu','cpu_threads':1,'clinical_data_used':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=FAMILIES);a=p.parse_args()
    if a.family:print(json.dumps(one(a.family)));return
    rows=[]
    for family in FAMILIES:
        r=subprocess.run([sys.executable,'-m','echonext.benchmark','--family',family],capture_output=True,text=True,check=True)
        rows.append(json.loads(r.stdout))
    Path('reports/synthetic_resource_benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'families_benchmarked':len(rows),'clinical_training':False,'max_peak_rss_bytes':max(r['peak_process_rss_bytes'] for r in rows)}))

if __name__=='__main__':main()
