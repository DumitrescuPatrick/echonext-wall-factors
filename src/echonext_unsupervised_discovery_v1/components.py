"""Post hoc secondary component enrichment, with multiplicity correction."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from echonext.schema import SECONDARY
from .data import verify_freeze,sha
from .associations import association

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();m=verify_freeze(a.run)
    if sha(a.metadata)!=m['details']['metadata_sha256']:raise ValueError('Metadata changed')
    targets={name:[] for name in SECONDARY}
    with a.metadata.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['split']=='val':
                for name in SECONDARY:
                    try: targets[name].append(float(row[name]))
                    except ValueError: targets[name].append(np.nan)
    results=[]
    for path in sorted(a.run.glob('*62001.assignments.npy')):
        if str(path.resolve()) not in m['files']:raise ValueError('Unfrozen assignment')
        labels=np.load(path,allow_pickle=False)
        for name,values in targets.items():
            y=np.array(values);keep=np.isin(y,[0,1])
            for result in association(labels[keep],y[keep]):results.append({'representation':path.name.split('_62001')[0],'outcome':name,'outcome_missing':int((~keep).sum()),**result})
    order=sorted([i for i,r in enumerate(results) if 'p_value' in r],key=lambda i:results[i]['p_value']);q=1.
    for rank,i in reversed(list(enumerate(order,1))):q=min(q,results[i]['p_value']*len(order)/rank);results[i]['q_value_BH']=q
    a.report.write_text(json.dumps({'post_hoc_exploratory':True,'correction_scope':'All reported component-cluster tests, distinct from primary-SHD correction','results':results},indent=2));print('Aggregate secondary component associations saved.')
if __name__=='__main__':main()
