"""Freeze native IDEC assignments and SVDD quantile outliers before associations."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .data import verify_freeze,freeze,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=verify_freeze(a.run)
    if 'private' not in a.output.resolve().parts:raise PermissionError('Private destination required')
    a.output.mkdir(parents=True,exist_ok=False);files=[]
    for file in sorted(a.run.glob('idec_*.npz')):
        z=np.load(file,allow_pickle=False)['val']; state=torch.load(file.with_suffix('.pt'),map_location='cpu',weights_only=True);centers=state['centers'].numpy();labels=((z[:,None]-centers)**2).sum(-1).argmin(1)
        dest=a.output/(file.stem.replace('idec','nativeidec')+'.assignments.npy');np.save(dest,labels);files.extend([dest,file,file.with_suffix('.pt')])
    for file in sorted(a.run.glob('svdd_*.anomaly.npz')):
        scores=np.load(file,allow_pickle=False);labels=(scores['validation_scores']>=scores['training_cutoff']).astype(int);dest=a.output/(file.name.replace('.anomaly.npz','.assignments.npy'));np.save(dest,labels);files.extend([dest,file])
    if not files:raise ValueError('No frozen native patterns available')
    details={**m['details'],'origin_freeze_sha256':sha(a.run/'FREEZE.json'),'pattern_definitions':'Native IDEC nearest Student-t center and Deep SVDD >= fitting 95th-percentile score, no outcomes used'}
    freeze(a.output,files+[a.run/'FREEZE.json',Path(__file__)],details);print('Native IDEC/SVDD pattern definitions frozen; outcome evaluation remains separate.')
if __name__=='__main__':main()
