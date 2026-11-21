"""Validation metrics and tie-safe operating points. No test-data entry point."""
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss,roc_curve
from sklearn.linear_model import LogisticRegression

def classification(y,p,threshold):
    y=np.asarray(y,int);d=np.asarray(p)>=threshold
    return {'sensitivity':float(d[y==1].mean()),'specificity':float((~d[y==0]).mean()),'flagged_fraction':float(d.mean())}

def threshold_for(y,p,kind,target):
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    if kind=='sensitivity':
        eligible=np.flatnonzero(tpr>=target);j=eligible[np.argmax(thresholds[eligible])]
    elif kind=='specificity':
        eligible=np.flatnonzero(1-fpr>=target);best=tpr[eligible].max();ids=eligible[tpr[eligible]==best];j=ids[np.argmax(thresholds[ids])]
    else:raise ValueError('Unrecognized operating-point type')
    return float(thresholds[j])

def metrics(y,p):
    y=np.asarray(y,int);p=np.clip(np.asarray(p,float),1e-7,1-1e-7)
    fit=LogisticRegression(C=1e6,max_iter=1000).fit(logit(p)[:,None],y)
    bins=np.minimum((p*10).astype(int),9)
    ece=sum(np.mean(bins==b)*abs(p[bins==b].mean()-y[bins==b].mean()) for b in range(10) if np.any(bins==b))
    return {'n':len(y),'positive':int(y.sum()),'auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
        'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p)),
        'calibration_intercept':float(fit.intercept_[0]),'calibration_slope':float(fit.coef_[0,0]),'ece_10':float(ece),
        'mean_risk':float(p.mean()),'risk_variance':float(p.var())}

def operating_points(y,p):
    rows=[]
    for kind in ['sensitivity','specificity']:
        for target in [.9,.95]:
            t=threshold_for(y,p,kind,target)
            rows.append({'target_kind':kind,'target':target,'threshold':t,**classification(y,p,t),'threshold_source':'same validation predictions; descriptive apparent point'})
    return rows

def sigmoid_calibration(y,logits):
    model=LogisticRegression(C=1e6,max_iter=1000).fit(np.asarray(logits)[:,None],y)
    return {'intercept':float(model.intercept_[0]),'slope':float(model.coef_[0,0])}

def apply_calibration(logits,c):return expit(c['intercept']+c['slope']*np.asarray(logits))
