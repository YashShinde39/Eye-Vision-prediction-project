# Limitations and Ethics

## Limitations

### Dataset

| Limitation | Detail |
|---|---|
| Small sample size | n = 81 rows; metrics should be treated as illustrative, not definitive |
| Self-reported features | Screen time, symptoms, and ergonomics are reported by participants and may contain recall or social-desirability bias |
| Limited diversity | Population is drawn from a constrained demographic; results may not generalise |
| Missing clinical labels | `vision_label` is empty in the raw CSV; all targets are proxy labels derived by a heuristic risk score |

### Labels

The `vision_status` target is constructed from a hand-crafted risk score that aggregates:

- Daily screen hours and session length
- Break frequency
- Outdoor time and viewing distance
- Sleep quality
- Headache and eyestrain frequency

Because the **same features drive both the label and the predictors**, the model may appear to perform better than it actually does on truly independent data. This is a **label-leakage / circularity risk** that is explicitly checked in the notebook's leakage analysis section but cannot be fully eliminated without clinically validated labels.

### Model

- With only 81 observations, a random train/test split can be highly variable. Stratified cross-validation is used throughout to mitigate this, but reported metrics still carry wide confidence intervals.
- XGBoost, even with shallow trees and regularisation, can overfit on a dataset this small.
- Calibrated probabilities are approximate; treat them as ordinal risk signals rather than precise probabilities.

---

## Ethical Considerations

### Non-Diagnostic

> **This tool is not a medical device.** It does not diagnose, screen for, or treat any eye condition or disorder. All outputs are exploratory risk indicators based on self-reported behaviour data. Users experiencing visual symptoms should consult a qualified eye-care professional.

### Fairness

- Model performance is evaluated separately across **age bands** (`<20`, `20–35`, `>35`), **gender**, and **device type** to detect disparate recall or precision.
- The dataset does not currently have sufficient representation to make robust fairness claims. Results should not be generalised to under-represented groups.

### Privacy

- No personally identifiable information (PII) is stored or published. The CSV contains anonymous survey responses.
- Any future deployment should ensure data minimisation and obtain informed consent.

### Transparency

- All label-construction assumptions (heuristic weights, thresholds, proxy-label logic) are documented in [`docs/data-acquisition-and-preprocessing.md`](data-acquisition-and-preprocessing.md) and the notebook.
- Model selection rationale and confounder-handling strategies are in [`docs/model-design-and-justification.md`](model-design-and-justification.md).

### Recommendations for Responsible Use

1. Replace proxy labels with validated clinical outcomes (e.g., CVS questionnaire scores or optometrist-measured acuity) before any real-world deployment.
2. Collect a larger, more demographically diverse dataset.
3. Pre-register severity thresholds to avoid post-hoc label tuning.
4. Use nested cross-validation with calibrated, group-aware evaluation for any production model.
5. Include a clear "when to seek care" disclaimer in any user-facing interface.
