# Data Acquisition and Preprocessing (Survey Design, Cleaning, Encoding)

Goal
- Collect consistent, privacy-safe self-reported data; clean and encode for ML on vision risk/severity.

1) Survey design (10–12 mins)
- Demographics: age (years), gender (female/male/prefer_not_say).
- Device context: device_type (mobile/laptop/tablet/tv), daily_hours (0–18h), session_length (minutes per typical session), breaks (count/day).
- Visual ergonomics: viewing_distance (cm), screen_height (below_eye/eye_level/above_eye), lighting (dim/normal/bright), brightness (low/medium/high), font_size (small/medium/large), dark_mode (yes/no).
- Behavior/health: outdoor_time (h/day), sleep_quality (1–5), headache_freq (1–5), eyestrain_freq (1–5).
- Outcomes (recommended): validated CVS symptom score or clinical screen. If unavailable, use proxy label (see below).
- Consent/privacy: brief notice, purpose, optionality, anonymous or pseudonymous ID.

Validation in form
- Required: age, device_type, daily_hours, session_length, breaks, viewing_distance, lighting, brightness, font_size, dark_mode, sleep_quality, headache_freq, eyestrain_freq.
- Ranges: age [10, 90], daily_hours [0, 18], session_length [5, 300], breaks [0, 60], outdoor_time [0, 10], viewing_distance [10, 100], Likert [1, 5].
- Normalization: lowercase strings; restrict options to enumerations.
- Soft logic: if device_type in {laptop/tablet} then typical viewing_distance 30–70 cm; flag outliers but allow.

2) Data schema (columns)
- age:int, gender:str, device_type:str, daily_hours:float, session_length:float, breaks:int
- font_size:str, brightness:str, dark_mode:str, outdoor_time:float
- viewing_distance:int, screen_height:str, lighting:str
- sleep_quality:int, headache_freq:int, eyestrain_freq:int
- Optional: outcome variables (e.g., cvs_score:int), vision_label:str (if clinically derived)

3) Cleaning rules
- Drop columns with >95% missing (e.g., milk_consumption_ml).
- Trim/normalize categorical text; map yes/no to {1,0}.
- Handle duplicates: drop exact duplicates; keep first for same [age, gender, device_type, daily_hours, session_length].
- Winsorize numeric at [1st, 99th] percentiles; hard clip by business ranges above.
- Imputation:
  - Numeric: median
  - Categorical: most frequent
  - Avoid imputing outcome labels; exclude rows without labels when training supervised models.
- Train/validation/test: stratified by binary target; recommend 60/20/20 for n>300; for n≈80 use stratified K-fold (k=5–10) and reserve a tiny holdout only if necessary.

4) Encoding scheme
- Ordinal encodings (domain order):
  - font_size: small(0) < medium(1) < large(2)
  - brightness: low(0) < medium(1) < high(2)
  - screen_height: above_eye(2) > eye_level(1) > below_eye(0) [risk-ordered]
- Binary: dark_mode yes→1, no→0; gender one-hot or binary if {female,male}.
- Nominal one-hot: device_type, lighting
- Standardize numeric for linear models: age, daily_hours, session_length, breaks, outdoor_time, viewing_distance, sleep_quality, headache_freq, eyestrain_freq, engineered features.

5) Feature engineering (examples)
- sessions_per_day = max(daily_hours*60 / session_length, 0), clipped [0, 100]
- breaks_per_hour = breaks / max(daily_hours, 0.25), clipped [0, 12]
- long_session = 1[session_length ≥ 90]
- short_distance = 1[viewing_distance < 35]
- outdoor_ratio = outdoor_time / max(daily_hours, 0.25), clipped [0, 2]
- high_brightness_flag = 1[brightness=high and dark_mode=0]

6) Labels
- Preferred: validated symptom scale (e.g., CVS questionnaire) or clinical screen; define binary and severity thresholds pre-registered.
- Interim proxy (only if needed): risk score combining daily_hours, long sessions, few breaks, short distance, poor sleep, high headaches/eyestrain, low outdoor_time; map to {normal, mild, moderate, severe}. Report as exploratory.

7) Class imbalance and evaluation
- Use stratified splits/K-fold; class_weight=“balanced” where applicable.
- Metrics: PR-AUC (impaired), macro F1, calibration (Brier), confusion matrix.

8) Reproducibility
- Fix random seeds; log preprocessing parameters; version datasets; export fitted encoders/scalers.