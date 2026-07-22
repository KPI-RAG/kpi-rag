# Data Setup — Required Before Running DatasetBuilder

I (Claude) could not perform this step myself — no network access to Hugging
Face from this environment. One team member needs to do this manually.

## 1. Download TelecomTS

```bash
pip install datasets huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='AliMaatouk/TelecomTS', repo_type='dataset',
                   local_dir='./data/telecomts_raw')
"
```

## 2. CRITICAL — confirm the real schema before running dataset_builder.py

`dataset_builder.py` currently assumes placeholder field names based on the
proposal's description (Section 6.3), NOT a verified real schema dump. Before
running it on real data:

```python
import json
with open("data/telecomts_raw/<first_file>.jsonl") as f:
    first_record = json.loads(f.readline())
print(json.dumps(first_record, indent=2))
```

Check specifically:
- [ ] Are the 16 numerical KPI channel names exactly as listed in `NUMERICAL_CHANNELS`
      in `dataset_builder.py`, or different (e.g., abbreviated, nested under a
      sub-key)?
- [ ] Is there a top-level field identifying which of the 33 source recording
      sessions a row belongs to, or is that only implicit in the filename?
- [ ] What is the exact field name and format for the anomaly condition type
      used for stratification (`anomaly_type` is a placeholder guess)?
- [ ] Confirm `UL_Protocol` / `DL_Protocol` field names and exact string values
      (the code assumes "TCP"/"UDP"/"None" — confirm capitalization).
- [ ] Confirm the `EXCLUDED_FIELDS` set in `dataset_builder.py` actually
      matches the real field names for `statistics`, `anomalies`, `labels`,
      `QnA`, `description`, and `troubleshooting_tickets` — these must never
      leak into the feature vector.

Update the placeholder names in `dataset_builder.py` to match, then re-run
the sanity test (see `tests/`) before trusting the output on real data.

## 3. Run

```bash
cd src/preprocessing
python dataset_builder.py
```

Outputs `train_X.npy`, `train_y.npy`, `test_X.npy`, `test_y.npy`, and
`split_meta.json` into `data/`.
