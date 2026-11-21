"""Patient-bootstrap centroid/density stability, independent of clinical outcomes."""
import argparse,json
from pathlib import Path
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score
from .data import verify_freeze

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--replicates',type=int,default=100);a=p.parse_args();m=verify_freeze(a.run);out=[]
    for file in sorted(a.run.glob('*62001.npz')):
        if str(file.resolve()) not in m['files']:raise ValueError('Unfrozen cache')
        z=np.load(file,allow_pickle=False); model=joblib.load(file.with_suffix('.joblib'));rng=np.random.default_rng(62001)
        reference=model['estimator'].predict(model['scaler'].transform(z['monitor']));aris=[]
        for b in range(a.replicates):
            i=rng.integers(0,len(z['fit']),len(z['fit']));scale=clone(model['scaler']).fit(z['fit'][i]);est=clone(model['estimator']);est.set_params(random_state=62001+b);est.fit(scale.transform(z['fit'][i]));pred=est.predict(scale.transform(z['monitor']));aris.append(float(adjusted_rand_score(reference,pred)))
        out.append({'representation':file.stem,'replicates':len(aris),'ARI_median':float(np.median(aris)),'ARI_2.5_percentile':float(np.quantile(aris,.025)),'ARI_97.5_percentile':float(np.quantile(aris,.975))})
    a.report.write_text(json.dumps({'scope':'Fixed selected K and method; whole fitting patients bootstrapped, frozen representations; does not include encoder retraining or K-selection uncertainty','results':out},indent=2));print('Aggregate patient-bootstrap stability saved.')
if __name__=='__main__':main()
