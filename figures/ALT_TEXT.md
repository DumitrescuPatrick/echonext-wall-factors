# Figure alt text

## Figure_1_Study_Design

Flow diagram showing source supervised ResNet training on 72,475 ECGs from 26,218 patients; downstream fitting in 23,597 patients, monitor calibration in 2,621, and evaluation in 4,626 previously explored validation patients. The validation cohort contributed to upstream selection; no independent confirmation is shown.

## Figure_2_Incremental_AUROC

Forest plot of six paired AUROC increments for logistic and gradient-boosting models after adding pooled factor 2, temporal factor 5, or both. Joint-factor intervals cross zero in both families. The small temporal-only logistic interval is positive, whereas the corresponding gradient-boosting interval crosses zero.

## Figure_3_Calibration

Calibration plots compare observed wall-thickness flag fractions with mean predicted probabilities in validation deciles for each reference and joint-factor model. Curves deviate from the ideal diagonal; they are descriptive and do not show new validation recalibration.

## Figure_4_Subgroups

Log-scale forest plot of mutually adjusted pooled and temporal factor odds ratios across all patients and predefined demographic, rate, conduction and care-setting subgroups. Estimates vary, with greater uncertainty in some groups; asterisks mark sparse or ill-conditioned fits. No interaction or independent-validation claim is made.

## Figure_5_Recording_and_Stability

Two panels contrast input sensitivity with estimator stability. A synchronized 100-millisecond shift changes pooled and temporal scores by RMS values of 0.431 and 0.645 training standard deviations. Separately, patient-bootstrap PCA refits on frozen features yield high matched monitoring-score correlations. The encoder was not retrained.

Alt text is embedded in DOCX figure descriptions and SVG title/description elements. Separate text is supplied for portal entry. The PDF is a visual proof, not a fully tagged accessible PDF.
