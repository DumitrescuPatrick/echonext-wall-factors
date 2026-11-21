"""Frozen-pattern morphology/confounding summaries; no individual exports."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .data import verify_freeze,sha

NUMERIC=['age_at_ecg','ventricular_rate','atrial_rate','pr_interval','qrs_duration','qt_corrected','acquisition_year']
CATEGORICAL=['sex','race_ethnicity','location_setting']

def number(value):
    try:
        x=float(value); return x if np.isfinite(x) else np.nan
    except (ValueError,TypeError): return np.nan

def summarize(labels,metadata):
    output=[]
    for k in np.unique(labels):
        keep=labels==k; n=int(keep.sum())
        if n<10: output.append({'cluster':int(k),'suppressed':True}); continue
        entry={'cluster':int(k),'patients':n,'numeric':{},'categorical':{}}
        for name in NUMERIC:
            values=np.array([number(r[name]) for r in metadata]); x=values[keep]; finite=x[np.isfinite(x)]
            overall=values[np.isfinite(values)]; sd=overall.std()
            if len(finite)<10: entry['numeric'][name]={'suppressed':True}; continue
            entry['numeric'][name]={'observed':len(finite),'mean':float(finite.mean()),'median':float(np.median(finite)),'standardized_mean_difference_from_population':float((finite.mean()-overall.mean())/sd) if sd>0 else None,'missing_fraction':float(np.isnan(x).mean())}
        for name in CATEGORICAL:
            values=np.array([r[name] or 'missing' for r in metadata]); counts={}
            for category in sorted(set(values)):
                count=int((values[keep]==category).sum())
                # Suppress both small cells and complements.
                counts[category]=count/n if min(count,n-count)>=10 else 'suppressed'
            entry['categorical'][name]=counts
        output.append(entry)
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    m=verify_freeze(a.run)
    if sha(a.metadata)!=m['details']['metadata_sha256']:raise ValueError('Metadata changed')
    rows=[]
    with a.metadata.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['split']=='val': rows.append({name:row.get(name,'') for name in NUMERIC+CATEGORICAL})
    result={}
    for file in sorted(a.run.glob('*62001.assignments.npy')):
        if str(file.resolve()) not in m['files']:raise ValueError('Unfrozen assignments')
        result[file.stem]=summarize(np.load(file,allow_pickle=False),rows)
    a.report.write_text(json.dumps({'exploratory':True,'scope':'Primary-seed validation morphology and demographics; observational, not proof of independence','patterns':result},indent=2))
    print('Aggregate morphology/confounding audit saved.')
if __name__=='__main__':main()
