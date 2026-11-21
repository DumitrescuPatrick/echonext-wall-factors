import unittest
import numpy as np
from echonext_wall_factor_validation_v1.run import paired,metrics,subgroups

class Statistics(unittest.TestCase):
    def test_identical_predictions_zero_paired_interval(self):
        y=np.tile([0,1],30);p=np.linspace(.05,.9,60)
        for value in paired(y,p,p,reps=30).values():
            self.assertEqual(value['difference'],0)
            self.assertEqual(value['paired_95_CI'],[0,0])
    def test_metric_direction_and_reproducibility(self):
        y=np.tile([0,1],30);a=np.full(60,.5);b=np.where(y,.9,.1)
        one=paired(y,a,b,reps=30);self.assertEqual(one,paired(y,a,b,reps=30))
        self.assertGreater(one['AUROC']['difference'],0);self.assertLess(one['Brier']['difference'],0)
        self.assertEqual(metrics(y,b)['sensitivity'],1)
    def test_subgroup_cut_uses_fit_only_and_missing_not_low(self):
        def row(year,rate,qrs):return dict(age_at_ecg='60',qrs_duration=qrs,ventricular_rate=rate,acquisition_year=year,sex='female',location_setting='outpatient')
        groups,cut=subgroups([row('2020','',''),row('2021','70','90')],[row('2000','70','90'),row('2002','70','90')])
        self.assertEqual(cut,2001);self.assertFalse(groups['rate:<60'][0]);self.assertFalse(groups['QRS:<120'][0]);self.assertFalse(groups['year:earlier'].any())

if __name__=='__main__':unittest.main()
