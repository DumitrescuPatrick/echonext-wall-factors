"""Resumable local ResNet fitting; no official-test loader or evaluation path."""
import argparse,hashlib,json,math,os,resource,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from .data import read_metadata,validate_splits,open_waveforms,canonical_batch,patient_weights
from .stage2_models import ResNet1D
from .stage2_metrics import metrics

OUT=Path('reports/stage2_v1');PRIVATE=Path('private/stage2_v1');CONFIG=Path('config/stage2_v1/protocol.json')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def dump(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def candidates(c):return [dict(a,**b) for a in c['architectures'] for b in c['regularization_candidates']]

def protocol():
    c=json.loads(CONFIG.read_text());receipt=OUT/'fit_source_manifest.json'
    if not receipt.exists():raise ValueError('Prepare source lock before training')
    for p,h in json.loads(receipt.read_text()).items():
        if sha(p)!=h:raise ValueError('Frozen fitting source changed')
    return c

def prepare():
    if (OUT/'fit_source_manifest.json').exists():raise FileExistsError('Fitting source already frozen')
    paths=[CONFIG,Path(__file__),Path(__file__).with_name('stage2_models.py'),Path(__file__).with_name('stage2_metrics.py'),Path(__file__).with_name('data.py')]
    paths+=list(Path('tests/stage2_v1').glob('*.py'))
    dump(OUT/'fit_source_manifest.json',{str(p):sha(p) for p in paths})

def development_inputs(root):
    md,_,_=read_metadata(root/'echonext_metadata_100k.csv');validate_splits(md)
    result={}
    for s in ('train','val'):
        a=open_waveforms(root/f'EchoNext_{s}_waveforms.npy');y=np.array(md[s]['labels'],np.float32)[:,0]
        if len(a)!=len(y) or not np.isfinite(y).all():raise ValueError('Development alignment/target mismatch')
        result[s]=(a,y)
    weights=patient_weights(md['train']['patients'],np.arange(len(result['train'][1])))
    result['weights']=weights/weights.mean()
    return result

def predict_logits(model,waveforms,batch=128):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(waveforms),batch):
            x=torch.from_numpy(canonical_batch(waveforms[start:start+batch]));result.append(model(x).numpy())
    return np.concatenate(result)

def selected_candidate(c):
    rows=[]
    for j in range(len(candidates(c))):
        rows.append(json.loads((OUT/f'tune_{j}_complete.json').read_text()))
    best=max(r['best_metrics']['auc'] for r in rows)
    eligible=[r for r in rows if r['best_metrics']['auc']>=best-1e-4]
    return min(eligible,key=lambda r:(r['best_metrics']['brier'],r['candidate_index']))

def fit(root,mode,index,seed=None):
    c=protocol();grid=candidates(c)
    if mode=='tune':
        chosen=index;seed=c['tuning_seed'];run=f'tune_{index}'
    else:
        if seed not in c['final_seeds']:raise ValueError('Unplanned final seed')
        chosen=selected_candidate(c)['candidate_index'];run=f'seed_{seed}'
    cfg=grid[chosen];folder=PRIVATE/run;folder.mkdir(exist_ok=True,mode=0o700)
    complete=OUT/(run+'_complete.json')
    if complete.exists():
        r=json.loads(complete.read_text())
        if sha(folder/'best.pt')!=r['best_checkpoint_sha256']:raise ValueError('Completed checkpoint changed')
        return
    if __import__('shutil').disk_usage('.').free<10*1024**3:raise RuntimeError('Insufficient free disk')
    torch.set_num_threads(c['cpu_threads']);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    fit_cpu_start=time.process_time();prior_cpu_seconds=0.
    inputs=development_inputs(root);wave,y=inputs['train'];vw,vy=inputs['val'];weights=inputs['weights']
    model=ResNet1D(cfg['width']);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,c['max_epochs'],eta_min=cfg['learning_rate']*.1)
    history=[];start_epoch=0;bad=0;best_auc=-1;best_brier=math.inf;best_epoch=0
    resume=folder/'last.pt'
    if resume.exists():
        z=torch.load(resume,map_location='cpu',weights_only=True)
        model.load_state_dict(z['model']);optimizer.load_state_dict(z['optimizer']);scheduler.load_state_dict(z['scheduler'])
        prior_cpu_seconds=z.get('fit_cpu_seconds',0.);start_epoch=z['epoch'];bad=z['bad'];best_auc=z['best_auc'];best_brier=z['best_brier'];best_epoch=z['best_epoch'];history=z['history']
    run_start=time.perf_counter()
    for epoch in range(start_epoch,c['max_epochs']):
        if epoch>=c['minimum_epochs'] and bad>=c['patience']:break
        model.train();order=np.random.default_rng(seed+1000003*(epoch+1)).permutation(len(y))
        t=time.perf_counter();cpu_t=time.process_time();total=0.;denom=0.
        for start in range(0,len(y),c['batch_size']):
            ix=order[start:start+c['batch_size']];x=torch.from_numpy(canonical_batch(wave[ix]))
            target=torch.from_numpy(y[ix]);w=torch.from_numpy(weights[ix]);optimizer.zero_grad()
            loss_rows=torch.nn.functional.binary_cross_entropy_with_logits(model(x),target,reduction='none')
            loss=(loss_rows*w).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float((loss_rows.detach()*w).sum());denom+=float(w.sum())
        logits=predict_logits(model,vw);score=metrics(vy,expit(logits));significant=score['auc']>best_auc+c['min_auc_delta']
        save=significant or (abs(score['auc']-best_auc)<=c['min_auc_delta'] and score['brier']<best_brier)
        if save:
            best_auc=score['auc'];best_brier=score['brier'];best_epoch=epoch+1
            torch.save({'state_dict':model.state_dict(),'width':cfg['width'],'seed':seed},folder/'best.pt')
            np.savez_compressed(folder/'validation_predictions.npz',y=vy,logits=logits)
        bad=0 if significant else bad+1;scheduler.step()
        row={'epoch':epoch+1,'train_weighted_bce':total/denom,'validation_auc':score['auc'],'validation_brier':score['brier'],'seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu_t,'bad_epochs':bad}
        history.append(row)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'bad':bad,'best_auc':best_auc,'best_brier':best_brier,'best_epoch':best_epoch,'history':history,'fit_cpu_seconds':prior_cpu_seconds+time.process_time()-fit_cpu_start},resume)
        dump(OUT/(run+'_history.json'),history)
        print(json.dumps({'run':run,**row}),flush=True)
    with np.load(folder/'validation_predictions.npz') as z:best_metrics=metrics(z['y'],expit(z['logits']))
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if __import__('sys').platform!='darwin':rss*=1024
    receipt={'run':run,'candidate_index':chosen,'configuration':cfg,'seed':seed,'epochs_completed':len(history),'best_epoch':best_epoch,'best_metrics':best_metrics,'n_train':len(y),'n_validation':len(vy),'trainable_parameters':sum(p.numel() for p in model.parameters()),'last_session_seconds':time.perf_counter()-run_start,'total_epoch_seconds':sum(r['seconds'] for r in history),'total_cpu_seconds':prior_cpu_seconds+time.process_time()-fit_cpu_start,'peak_rss_bytes':rss,'early_stopping':'official validation AUC','test_access':False,'best_checkpoint_sha256':sha(folder/'best.pt'),'private_prediction_sha256':sha(folder/'validation_predictions.npz')}
    dump(complete,receipt)
    print(json.dumps({'run_complete':run,'epochs':len(history),'best_epoch':best_epoch,'validation_auc':best_metrics['auc']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','tune','final']);p.add_argument('--index',type=int,default=0);p.add_argument('--seed',type=int);p.add_argument('--data-dir',type=Path,default=Path.home()/'Downloads');a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:fit(a.data_dir,a.mode,a.index,a.seed)
