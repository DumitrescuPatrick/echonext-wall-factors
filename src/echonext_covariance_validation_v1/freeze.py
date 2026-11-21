"""Create a new immutable reference to existing definitions; never rewrite old freezes."""
import json
from pathlib import Path
from echonext_unsupervised_discovery_v1.data import sha,verify_freeze,freeze

PRIVATE=Path('private/covariance_validation_v1')
CONFIG=Path('config/covariance_validation_v1/protocol.json')
BASE=Path('private/unsupervised_discovery_v1/classical_v1')

def main():
    previous=verify_freeze(BASE)
    if PRIVATE.exists():raise FileExistsError('Preserve validation project version')
    PRIVATE.mkdir(parents=True,mode=0o700)
    paths=[BASE/'FREEZE.json',CONFIG,Path('src/echonext_unsupervised_discovery_v1/features.py')]
    for method in ['spd','waveform','hilbert']:
        for seed in [62001,62002,62003,62004,62005]:paths.extend([BASE/f'{method}_{seed}.joblib',BASE/f'{method}_{seed}.assignments.npy'])
    paths.extend([BASE/'spd_62001.npz',BASE/'waveform_62001.npz'])
    report=Path('reports/stage2_v1/tune_3_complete.json');r=json.loads(report.read_text());checkpoint=Path('private/stage2_v1/tune_3/best.pt');pred=Path('private/stage2_v1/tune_3/validation_predictions.npz')
    if sha(checkpoint)!=r['best_checkpoint_sha256'] or sha(pred)!=r['private_prediction_sha256']:raise ValueError('ResNet integrity mismatch')
    paths.extend([report,checkpoint,pred,Path('config/stage2_v1/protocol.json')])
    freeze(PRIVATE,paths,{'meaning':'Snapshot of existing features, group assignments and ResNet; no re-clustering/retraining','metadata_sha256':previous['details']['metadata_sha256'],'test_opened':False,'ResNet_score_status':'Training scores would be in-sample; validation scores used during checkpoint selection; honest OOF scores unavailable'})
    Path('reports/covariance_validation_v1/FREEZE_RECEIPT.json').write_text(json.dumps({'existing_definitions_frozen':True,'original_groups_changed':False,'linked_files':len(paths),'private_freeze_sha256':sha(PRIVATE/'FREEZE.json'),'ResNet_OOF_scores_available':False,'test_opened':False},indent=2));print('Definitions/groups/ResNet linked in a new private freeze; originals unchanged.')
if __name__=='__main__':main()
