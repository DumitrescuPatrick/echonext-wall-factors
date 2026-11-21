"""Deterministic descriptors, log-Euclidean SPD covariance, Gaussian RKHS map."""
import numpy as np

def waveform_features(x):
    if x.ndim!=3 or x.shape[1]!=12 or not np.isfinite(x).all(): raise ValueError('Invalid waveform')
    return np.concatenate([x.mean(-1),x.std(-1),np.ptp(x,axis=-1),np.sqrt(np.mean(np.diff(x,axis=-1)**2,axis=-1))],axis=1)

def spd_features(x):
    a=x.astype('float64'); a=a-a.mean(-1,keepdims=True); c=a@a.transpose(0,2,1)/max(1,x.shape[-1]-1)
    scale=np.trace(c,axis1=1,axis2=2)/12; c=.9*c/np.maximum(scale[:,None,None],1e-10)+.1*np.eye(12)
    eigen,u=np.linalg.eigh(c)
    if (eigen<=0).any(): raise ValueError('Nonpositive covariance')
    logs=(u*np.log(eigen)[:,None,:])@u.transpose(0,2,1); i,j=np.triu_indices(12)
    return np.concatenate([logs[:,i,j]*np.where(i==j,1,np.sqrt(2)),np.log(np.maximum(scale[:,None],1e-10))],1).astype('float32')

class HilbertMap:
    def __init__(self,d,k=128,seed=62001):
        rng=np.random.default_rng(seed); self.omega=rng.normal(size=(d,k))/np.sqrt(d); self.phase=rng.uniform(0,2*np.pi,k)
    def transform(self,x): return (np.sqrt(2/len(self.phase))*np.cos(x@self.omega+self.phase)).astype('float32')
