# MTA Kaggle fixtures

These compact fixtures contain deterministic subsets of the user-supplied
Kaggle downloads. They let CI exercise real dataset schemas without committing
roughly 2 GB of source CSVs.

Coverage:

- Instacart order products: binary classification with `reordered` as target.
- IEEE-CIS fraud: binary classification with `isFraud` as target, including
  missing numeric values and categorical features.
- Walmart M5 sales: regression using the six previous daily-sales columns to
  predict the next day.
- IEEE test transactions: verifies that an unlabeled competition test file is
  rejected as training data.

Quality gates:

- Instacart uses a deterministic 80/20 hold-out from the fixture. It trains
  only on the 80% partition, runs actual model inference on the unseen rows,
  and requires held-out accuracy to beat the majority-class baseline by at
  least one percentage point, positive-class F1 of at least 0.60, balanced
  accuracy of at least 0.52, and ROC AUC of at least 0.55.
- IEEE fraud uses the same held-out approach and requires accuracy to beat the
  majority baseline by at least ten percentage points, F1 and balanced
  accuracy of at least 0.60, and ROC AUC of at least 0.65. Because the compact
  fixture is deliberately class-balanced and sampled sequentially, this gate
  is a regression guard rather than a production fraud-performance estimate.
- Walmart uses the same deterministic hold-out approach. Its inference MAE
  and RMSE must beat a naive predictor that returns the training-set mean for
  every row, and held-out R-squared must be positive.
- IEEE fraud trains and performs inference with an unseen categorical value.
  This verifies that new values do not silently become an arbitrary ordinal
  number or crash inference; they use the fitted unknown-category bucket.
- Focused preprocessing tests prove that categorical vocabularies, missing
  value fill values, and numeric scaling are fitted from training rows only.
  They also verify that minority classes receive a larger classification-loss
  weight than majority classes.

Run the quality gates and print their measured values with:

```bash
pytest -s tests/test_mta_v2_kaggle_automation.py -k held_out_inference
```

Run only the fast preprocessing safeguards:

```bash
pytest tests/test_mta_v2_kaggle_automation.py \
  -k 'preprocessing_uses_training or balanced_class_weights'
```

Run all committed fixture tests (schema, training, inference, and quality):

```bash
pytest -s tests/test_mta_v2_kaggle_automation.py
```

These are regression guards, not leaderboard claims: passing means the
implementation beats simple no-skill baselines on real held-out fixture rows.

Regenerate the fixtures after downloading the datasets:

```bash
python tests/fixtures/mta_kaggle/build_fixtures.py /path/to/dataset-directory
```

To validate the original full downloads as well as the committed fixtures:

```bash
MTA_KAGGLE_DATASET_DIR=/path/to/dataset-directory \
pytest tests/test_mta_v2_kaggle_automation.py -m kaggle
```
