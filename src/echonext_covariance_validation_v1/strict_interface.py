"""Gate independent scoring on all model ancestors, not just the fusion fold."""
import numpy as np
from sklearn.metrics import roc_auc_score,brier_score_loss
from .analysis import check_patient_separation

def independent_increment(y,reference,covariance,target_patients,learner_ancestry):
    """Arrays and raw linkage stay private; this function returns aggregates only.

    learner_ancestry must list ALL base/fusion/preprocessor/selector/calibrator
    learners producing either evaluation scores or meta-training features. Audited
    investigator-provided lineage is required; this cannot infer hidden data use.
    """
    if not learner_ancestry:raise ValueError('Complete learner ancestry is required')
    if len(set(target_patients))!=len(target_patients):raise ValueError('Independent evaluation requires one row per patient')
    for learner in learner_ancestry:
        if not learner.get('lineage_complete',False):raise ValueError('Incomplete learner history')
        required=['fit_patients','selection_patients','calibration_patients']
        if not all(name in learner for name in required):raise ValueError('Missing patient-use role')
        check_patient_separation(target_patients,learner['fit_patients'],learner['selection_patients'],learner['calibration_patients'])
    y=np.asarray(y);a=np.asarray(reference);b=np.asarray(covariance)
    if not (len(y)==len(a)==len(b)==len(target_patients)) or not np.isin(y,[0,1]).all() or len(np.unique(y))<2:raise ValueError('Invalid independent outcome population')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any((a<0)|(a>1)|(b<0)|(b>1)):raise ValueError('Invalid probabilities')
    return {'patients':len(y),'base_and_fusion_patient_separation_verified_from_supplied_lineage':True,'AUROC_difference':float(roc_auc_score(y,b)-roc_auc_score(y,a)),'Brier_difference':float(brier_score_loss(y,b)-brier_score_loss(y,a))}
