"""Physiology-specific claims require external validation provenance."""
import numpy as np
from scipy.signal import resample_poly,resample

def validated_segments(x,qrs_on,qrs_off,t_off,provenance):
    if not provenance.get('independent_expert_reference_validated') or not provenance.get('validation_metrics') or not provenance.get('source_hash'):raise ValueError('Validated independent fiducials required for QRS/ST-T-specific claims')
    if len(qrs_on)!=len(qrs_off) or len(qrs_off)!=len(t_off):raise ValueError('Fiducial lengths differ')
    qrs=[];stt=[]
    for a,b,c in zip(qrs_on,qrs_off,t_off):
        if not 0<=a<b<c<=x.shape[-1]:raise ValueError('Invalid beat boundaries')
        qrs.append(resample(x[:,a:b],128,axis=-1));stt.append(resample(x[:,b:c],256,axis=-1))
    return np.array(qrs),np.array(stt)

def axis_claim_allowed(kind,provenance):
    if kind=='provider_recorded_axis_category':return bool(provenance.get('dataset') and provenance.get('recorded_not_estimated'))
    if kind=='validated_QRS_area_angle':return bool(provenance.get('independent_reference_validation') and provenance.get('calibration_hash') and provenance.get('validated_fiducials'))
    return False
