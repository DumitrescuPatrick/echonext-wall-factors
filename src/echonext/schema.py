"""Official label names; reference measurements never enter model inputs."""
PRIMARY = 'shd_moderate_or_greater_flag'
SECONDARY = [
    'lvef_lte_45_flag', 'lvwt_gte_13_flag',
    'aortic_stenosis_moderate_or_greater_flag',
    'aortic_regurgitation_moderate_or_greater_flag',
    'mitral_regurgitation_moderate_or_greater_flag',
    'tricuspid_regurgitation_moderate_or_greater_flag',
    'pulmonary_regurgitation_moderate_or_greater_flag',
    'rv_systolic_dysfunction_moderate_or_greater_flag',
    'pericardial_effusion_moderate_large_flag',
    'pasp_gte_45_flag', 'tr_max_gte_32_flag']
LABELS = [PRIMARY] + SECONDARY
SPLITS = ('train', 'val', 'test', 'no_split')
WAVE_SHAPE = (1, 2500, 12)
SAMPLING_HZ = 250
