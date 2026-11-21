"""Finish authorized baseline work automatically after all six tuning trials."""
import json,os,subprocess,sys,time
from pathlib import Path

def main():
    out=Path('reports/stage2_v1')
    while len(list(out.glob('tune_*_complete.json')))<6:
        time.sleep(30)
    print(json.dumps({'tuning_finished':6,'starting_five_final_seeds':True}),flush=True)
    groups=[[53001,253001,453001],[153001,353001]];workers=[]
    for seeds in groups:
        code='from pathlib import Path\nfrom echonext.stage2_train import fit\n'+''.join(f"fit(Path('private/stage2_v1/development_cache'),'final',0,{seed})\n" for seed in seeds)
        workers.append(subprocess.Popen([sys.executable,'-c',code]))
    for p in workers:
        if p.wait()!=0:raise RuntimeError('Final seed worker failed; private resumable checkpoints retained')
    env=dict(os.environ);env['MPLCONFIGDIR']='/tmp/echonext-mpl'
    for module,args in [('stage2_report',[]),('stage2_finish',[]),('stage2_geometric',[]),('stage2_finish',['package'])]:
        subprocess.run([sys.executable,'-m','echonext.'+module]+args,check=True,env=env)

if __name__=='__main__':main()
