"""Sequential, private-cache discovery runner. No test arrays or outcomes loaded."""
import argparse, json, os, random, time, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import joblib
from .data import Waves, roles, sha, freeze, verify_freeze
from .models import Autoencoder, IDEC, SimCLR, DeepSVDD, augment, contrastive_loss
from .features import waveform_features, spd_features, HilbertMap

SEEDS=[62001,62002,62003,62004,62005]

def device(required='mps'):
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1': raise RuntimeError('Silent MPS CPU fallback is prohibited')
    if required=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable in this process. Run from native Apple Silicon Terminal; no CPU substitution.')
    return torch.device(required)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)

def embed(model,loader,dev,mode):
    values=[]; model.eval()
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(dev)
            if mode=='resnet': z=model.blocks(model.stem(x)).mean(-1)
            elif mode=='idec': z=model.ae.encoder(x)
            else: z=model.encoder(x)
            values.append(z.cpu().numpy())
    return np.concatenate(values)

def neural_fit(mode,train,monitor,dev,seed,epochs,batch,workers):
    seed_all(seed)
    model={'vae':lambda:Autoencoder(variational=True),'simclr':SimCLR,'svdd':DeepSVDD,'idec':IDEC}[mode]().to(dev)
    groups=train.groups[train.indices]; unique,freq=np.unique(groups,return_counts=True); weights=dict(zip(unique,1/freq))
    sampler=WeightedRandomSampler([weights[g] for g in groups],len(train),replacement=True)
    loader=DataLoader(train,batch_size=batch,sampler=sampler,num_workers=workers,pin_memory=False)
    stop=DataLoader(monitor,batch_size=batch,num_workers=workers,pin_memory=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    history=[]; best=float('inf'); best_state=None; patience=0
    # IDEC: deterministic AE pretraining followed by train-only KMeans initialization.
    if mode=='idec':
        for _ in range(5):
            model.train()
            for x,_ in loader:
                x=x.to(dev); optimizer.zero_grad(); loss=model.ae.loss(x,beta=0); loss.backward(); optimizer.step()
        z=embed(model,DataLoader(train,batch_size=batch,num_workers=workers),dev,mode)
        centers=KMeans(4,n_init=10,random_state=seed).fit(z).cluster_centers_
        with torch.no_grad(): model.centers.copy_(torch.tensor(centers,device=dev))
    if mode=='svdd':
        z=embed(model,DataLoader(train,batch_size=batch,num_workers=workers),dev,mode)
        c=torch.tensor(z.mean(0),device=dev); c=torch.where(c.abs()<.1,torch.where(c<0,-.1,.1),c)
        model.center.copy_(c)
    def loss_for(x,g):
        if mode=='vae': return model.loss(x)
        if mode=='simclr': return contrastive_loss(model(augment(x)),model(augment(x)),g)
        if mode=='svdd': return model(x).mean()
        # Target distribution uses entire fit set, not minibatch frequencies (see indexed loop below).
        raise RuntimeError('IDEC requires indexed full-training targets')
    for epoch in range(epochs):
        model.train(); total=torch.zeros((),device=dev); n=0
        if mode=='idec':
            ordered=DataLoader(train,batch_size=batch,num_workers=workers)
            model.eval(); qs=[]
            with torch.no_grad():
                for x,_ in ordered: qs.append(model(x.to(dev))[2].cpu())
            target=IDEC.target(torch.cat(qs)); model.train()
            # Patient weighted reconstruction/cluster loss via weighted sampling is replaced with one representative per patient upstream for IDEC.
            for start,(x,_) in enumerate(ordered):
                x=x.to(dev); p=target[start*batch:start*batch+len(x)].to(dev); optimizer.zero_grad(); loss=model.loss(x,p); loss.backward(); optimizer.step(); total+=loss.detach()*len(x); n+=len(x)
        else:
            for x,g in loader:
                x=x.to(dev); g=g.to(dev); optimizer.zero_grad(); loss=loss_for(x,g)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite objective')
                loss.backward(); optimizer.step(); total+=loss.detach()*len(x); n+=len(x)
        model.eval(); vt=torch.zeros((),device=dev); vn=0
        with torch.no_grad():
            for x,g in stop:
                x=x.to(dev); g=g.to(dev)
                if mode=='idec': loss=(model(x)[0]-x).square().mean() # Prespecified reconstruction monitor; no SHD.
                else: loss=loss_for(x,g)
                vt+=loss*len(x); vn+=len(x)
        value=float((vt/vn).cpu()); history.append({'epoch':epoch+1,'training_loss':float((total/n).cpu()),'monitor_loss':value})
        if value<best-1e-5:
            best=value; patience=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: patience+=1
        if patience>=5: break
    model.load_state_dict(best_state); return model,history

def clusters(zfit,zmonitor,zval,seed):
    scaler=StandardScaler().fit(zfit); a=scaler.transform(zfit); b=scaler.transform(zmonitor); c=scaler.transform(zval)
    candidates=[]
    for kind in ('kmeans','gmm'):
        for k in (2,3,4,6,8):
            estimator=(KMeans(k,n_init=10,random_state=seed) if kind=='kmeans' else GaussianMixture(k,covariance_type='diag',reg_covar=1e-5,n_init=3,random_state=seed)).fit(a)
            labels=estimator.predict(b)
            if len(np.unique(labels))<2 or len(np.unique(labels))>=len(b): continue
            score=silhouette_score(b,labels,sample_size=min(3000,len(b)),random_state=seed)
            candidates.append((score,kind,k,estimator,labels))
    if not candidates: raise RuntimeError('No noncollapsed clustering')
    score,kind,k,estimator,labels=max(candidates,key=lambda q:q[0])
    return {'scaler':scaler,'estimator':estimator},estimator.predict(c),{'silhouette':float(score),'davies_bouldin':float(davies_bouldin_score(b,labels)),'method':kind,'k':k,'candidates':[{'silhouette':float(t[0]),'method':t[1],'k':t[2]} for t in candidates]}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['mps','cpu'],default='mps'); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--workers',type=int,choices=[0,1,2],default=0)
    p.add_argument('--methods',nargs='+',choices=['waveform','spd','hilbert','resnet','vae','idec','simclr','svdd'],default=['waveform','spd','hilbert','resnet','vae','idec','simclr','svdd']); p.add_argument('--resnet-checkpoint',type=Path)
    args=p.parse_args(); dev=device(args.device)
    if args.device=='cpu' and any(m in args.methods for m in ('vae','idec','simclr','svdd','resnet')):
        raise RuntimeError('Clinical neural execution requires MPS in this GPU research protocol')
    if args.output.exists(): raise FileExistsError('Output must be a new private run directory')
    if 'private' not in args.output.resolve().parts: raise PermissionError('Patient-level caches must remain in private/')
    if args.epochs<1: raise ValueError('Positive epochs required')
    # Verify the existing source/cache manifest before any representation fitting.
    integrity_path=Path('reports/stage2_v1/development_cache_integrity.json')
    integrity=json.loads(integrity_path.read_text())
    for name,expected in integrity.items():
        path=args.root/name
        if path.stat().st_size!=expected['bytes'] or sha(path)!=expected['sha256']:
            raise ValueError('Development cache provenance mismatch')
    args.output.mkdir(parents=True)
    train=Waves(args.root,'train'); val=Waves(args.root,'val'); fit,monitor,reps=roles(train.patients)
    # Primary clustering and label-free validation: one deterministic ECG per patient.
    _,first=np.unique(np.array(train.patients)[monitor],return_index=True); monitor=monitor[np.sort(first)]
    fitdata=Waves(args.root,'train',fit); repdata=Waves(args.root,'train',reps); mondata=Waves(args.root,'train',monitor)
    loaders=[DataLoader(d,batch_size=args.batch_size,num_workers=args.workers,pin_memory=False) for d in (repdata,mondata,val)]
    records=[]; artifacts=[]; all_labels={}
    for method in args.methods:
        all_labels[method]=[]
        raw_features=None
        if method in ('waveform','spd','hilbert'):
            raw_features=[np.concatenate([(spd_features(x.numpy()) if method=='spd' else waveform_features(x.numpy())) for x,_ in loader]) for loader in loaders]
        for seed in SEEDS:
            stem=args.output/f'{method}_{seed}'; history=[]
            if method in ('waveform','spd','hilbert'):
                embeddings=raw_features
                if method=='hilbert':
                    scale=StandardScaler().fit(embeddings[0]); mapper=HilbertMap(48,seed=seed); embeddings=[mapper.transform(scale.transform(z)) for z in embeddings]
                    transform=stem.with_suffix('.transform.joblib'); joblib.dump((scale,mapper),transform); artifacts.append(transform)
            elif method=='resnet':
                from echonext.stage2_models import ResNet1D
                if args.resnet_checkpoint is None: raise ValueError('Explicit supervised reference checkpoint required')
                checkpoint=torch.load(args.resnet_checkpoint,map_location='cpu',weights_only=False); model=ResNet1D(checkpoint['width']); model.load_state_dict(checkpoint['state_dict']); model.to(dev)
                embeddings=[embed(model,l,dev,method) for l in loaders]
            else:
                model,history=neural_fit(method,repdata if method=='idec' else fitdata,mondata,dev,seed,args.epochs,args.batch_size,args.workers)
                checkpoint=stem.with_suffix('.pt'); torch.save(model.cpu().state_dict(),checkpoint); artifacts.append(checkpoint); model.to(dev)
                embeddings=[embed(model,l,dev,method) for l in loaders]
                if method=='svdd':
                    scores=((embeddings[2]-model.center.detach().cpu().numpy())**2).sum(1)
                    scorefile=stem.with_suffix('.anomaly.npz'); np.savez(scorefile,validation_scores=scores,training_cutoff=np.quantile(((embeddings[0]-model.center.detach().cpu().numpy())**2).sum(1),.95)); artifacts.append(scorefile)
            cache=stem.with_suffix('.npz'); np.savez(cache,fit=embeddings[0],monitor=embeddings[1],val=embeddings[2]); artifacts.append(cache)
            estimator,labels,metrics=clusters(*embeddings,seed); fitted=stem.with_suffix('.joblib'); joblib.dump(estimator,fitted); artifacts.append(fitted)
            assignments=stem.with_suffix('.assignments.npy'); np.save(assignments,labels); artifacts.append(assignments); all_labels[method].append(labels)
            historypath=stem.with_suffix('.history.json'); historypath.write_text(json.dumps(history,indent=2)); artifacts.append(historypath)
            records.append({'representation':method,'seed':seed,**metrics})
            (args.output/'progress.json').write_text(json.dumps(records,indent=2))
    summary={'runs':records,'validation_assignment_stability_ARI':{m:[float(adjusted_rand_score(v[i],v[j])) for i in range(len(v)) for j in range(i)] for m,v in all_labels.items()},'test_opened':False,'labels_used':False}
    report=args.output/'discovery.json'; report.write_text(json.dumps(summary,indent=2)); artifacts.append(report)
    source=list(Path(__file__).parent.glob('*.py')); metadata=args.root/'echonext_metadata_100k.csv'
    details={'seeds':SEEDS,'device':str(dev),'epochs_cap':args.epochs,'batch_size':args.batch_size,'workers':args.workers,'metadata_sha256':sha(metadata),'cache_integrity_reference':str(integrity_path.resolve()),'verified_cache_sha256':integrity,'waveform_alignment':'Provider row-order guarantee; no embedded identifiers','supervised_reference':'resnet' in args.methods,'fit_rows_sha256':__import__('hashlib').sha256(fit.tobytes()).hexdigest(),'resnet_checkpoint_sha256':sha(args.resnet_checkpoint) if args.resnet_checkpoint else None,'representation_scope':'selected methods only; no claims about omitted methods'}
    freeze(args.output,artifacts+source+[integrity_path],details)
    print('Discovery definitions frozen locally. No SHD outcomes or test data accessed.')

if __name__=='__main__': main()
