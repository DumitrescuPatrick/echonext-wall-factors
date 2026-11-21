"""Statistical utilities and explicit provenance gates; no test reader."""
import numpy as np
from scipy.stats import norm

def check_patient_separation(target_patients,fit_patients,selection_patients=(),calibration_patients=()):
    target=set(target_patients);used=set(fit_patients)|set(selection_patients)|set(calibration_patients)
    if target&used:raise ValueError('Target patients influenced base-model fitting, selection or calibration; independent increment claim prohibited')
    return True

def net_benefit(y,p,thresholds):
    y=np.asarray(y);p=np.asarray(p);t=np.asarray(thresholds)
    if not np.isin(y,[0,1]).all() or np.any((t<=0)|(t>=1)):raise ValueError('Invalid decision-curve inputs')
    positive=p[:,None]>=t[None,:]
    return np.mean(positive*y[:,None]-positive*(1-y[:,None])*t[None,:]/(1-t[None,:]),axis=0)

def change_interval(changed,n):
    z=norm.ppf(.975);p=changed/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,center-half)),float(min(1,center+half))]
