"""
DatasetBuilder — Layer 1: Preprocessing and Feature Engineering
Owner: P1

Implements exactly what's specified in the proposal (Section 6.3):
  - Loads TelecomTS at native scale (no normalization)
  - Builds the 582-dimensional feature vector per 128-timestep window
  - Splits at the RECORDING level (not window level) to prevent leakage
    from the 75% overlap inherent in stride-32 windowing
  - Explicitly excludes label-adjacent fields from the feature vector

USAGE:
    Requires the raw TelecomTS JSONL files (33 recording sessions) downloaded
    locally first — see docs/data_setup.md for the download step, which must
    be run by a team member (not available in this sandboxed environment).

    from dataset_builder import DatasetBuilder
    builder = DatasetBuilder(data_dir="../data/telecomts_raw")
    train_X, train_y, test_X, test_y, meta = builder.build()
"""

import json
import glob
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter


# Fields that must NEVER enter the feature vector — these either directly
# encode the label or leak future/label-adjacent information.
EXCLUDED_FIELDS = {
    "statistics", "anomalies", "labels", "QnA",
    "description", "troubleshooting_tickets",
}

# The 16 numerical KPI channels used for feature extraction (confirm exact
# field names against the actual downloaded schema before running — these
# names are per the proposal's Table in 6.3 and may need adjustment).
NUMERICAL_CHANNELS = [
    "RSRP", "UL_SNR", "DL_BLER", "UL_BLER", "DL_MCS", "UL_MCS",
    "UL_NPRB", "Estimated_UL_Buffer", "PRBs_DL_Current", "PRBs_UL_Current",
    "PRB_Utilization_DL", "PRB_Utilization_UL", "TX_Bytes", "RX_Bytes",
    "UL_NumberOfPackets", "DL_NumberOfPackets",
]
CATEGORICAL_CHANNELS = ["UL_Protocol", "DL_Protocol"]
PROTOCOL_VALUES = ["TCP", "UDP", "None"]

WINDOW_LEN = 128
STRIDE = 32
N_PATCHES = 8
PATCH_LEN = WINDOW_LEN // N_PATCHES  # 16


class DatasetBuilder:
    def __init__(self, data_dir: str, random_state: int = 42):
        self.data_dir = Path(data_dir)
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Step 1: Load raw recordings
    # ------------------------------------------------------------------
    def load_recordings(self):
        """
        Loads each of the 33 source JSONL files as one recording session.
        Returns a dict: {recording_id: raw_record_dict}
        """
        files = sorted(glob.glob(str(self.data_dir / "*.jsonl")))
        if not files:
            raise FileNotFoundError(
                f"No .jsonl files found in {self.data_dir}. "
                "Download TelecomTS first — see docs/data_setup.md."
            )
        recordings = {}
        for i, fpath in enumerate(files):
            with open(fpath) as f:
                recordings[i] = [json.loads(line) for line in f]
        print(f"Loaded {len(recordings)} recording sessions from {len(files)} files.")
        return recordings

    # ------------------------------------------------------------------
    # Step 2: Sliding window over a single recording's KPI time series
    # ------------------------------------------------------------------
    def sliding_windows(self, kpi_array: np.ndarray):
        """
        kpi_array: shape (T, 16) — raw KPI values over the full recording
        Returns: list of (start_idx, window) where window has shape (128, 16)
        """
        T = kpi_array.shape[0]
        windows = []
        for start in range(0, T - WINDOW_LEN + 1, STRIDE):
            windows.append((start, kpi_array[start:start + WINDOW_LEN]))
        return windows

    # ------------------------------------------------------------------
    # Step 3: 582-dim feature vector per window
    # ------------------------------------------------------------------
    def extract_features(self, window: np.ndarray, protocol_onehot: np.ndarray) -> np.ndarray:
        """
        window: shape (128, 16) raw KPI values, NO normalization applied
        protocol_onehot: shape (6,) — one-hot UL/DL protocol
        Returns: 582-dim feature vector
            64  precomputed statistics   (mean, std, min, max x 16 channels)
            256 patchwise scale stats    (8 patches x mean+std x 16 channels)
            256 first-order differences  (8 patches x mean+std x 16 channels)
            6   categorical encoding
        """
        # --- 64: precomputed statistics (mean, std, min, max per channel) ---
        stats = np.concatenate([
            window.mean(axis=0), window.std(axis=0),
            window.min(axis=0), window.max(axis=0),
        ])  # 4 x 16 = 64

        # --- 256: patchwise scale statistics ---
        patches = window.reshape(N_PATCHES, PATCH_LEN, -1)  # (8, 16, 16)
        patch_mean = patches.mean(axis=1).flatten()  # 8 x 16 = 128
        patch_std = patches.std(axis=1).flatten()    # 8 x 16 = 128
        patchwise_scale = np.concatenate([patch_mean, patch_std])  # 256

        # --- 256: first-order differences, patchwise mean+std ---
        diff = np.diff(window, axis=0)  # (127, 16)
        # pad to keep patch math clean: drop last patch's final row (127 vs 128)
        diff_padded = np.vstack([diff, diff[-1:]])  # (128, 16)
        diff_patches = diff_padded.reshape(N_PATCHES, PATCH_LEN, -1)
        diff_patch_mean = diff_patches.mean(axis=1).flatten()  # 128
        diff_patch_std = diff_patches.std(axis=1).flatten()    # 128
        diff_features = np.concatenate([diff_patch_mean, diff_patch_std])  # 256

        # --- 6: categorical encoding ---
        # protocol_onehot supplied by caller (3 classes x 2 protocols)

        vec = np.concatenate([stats, patchwise_scale, diff_features, protocol_onehot])
        assert vec.shape[0] == 582, f"Expected 582 dims, got {vec.shape[0]}"
        return vec

    def encode_protocol(self, ul_protocol: str, dl_protocol: str) -> np.ndarray:
        ul_vec = np.zeros(3)
        dl_vec = np.zeros(3)
        ul_vec[PROTOCOL_VALUES.index(ul_protocol) if ul_protocol in PROTOCOL_VALUES else 2] = 1
        dl_vec[PROTOCOL_VALUES.index(dl_protocol) if dl_protocol in PROTOCOL_VALUES else 2] = 1
        return np.concatenate([ul_vec, dl_vec])  # 6

    # ------------------------------------------------------------------
    # Step 4: Recording-level stratified split (NOT window-level)
    # ------------------------------------------------------------------
    def recording_level_split(self, recording_labels: dict, test_size: float = 0.2):
        """
        recording_labels: {recording_id: dominant_anomaly_condition_type}
        Splits at the RECORDING level so that overlapping windows from the
        same session never appear in both train and test — this is the
        leakage fix specified in the proposal (Section 6.3).
        """
        ids = list(recording_labels.keys())
        strata = [recording_labels[i] for i in ids]

        counts = Counter(strata)
        rare = [c for c, n in counts.items() if n < 2]
        if rare:
            print(f"WARNING: {len(rare)} condition type(s) have <2 recordings "
                  f"({rare}) — stratification will fail or be unstable for these. "
                  "Confirm with the team whether to merge rare strata before splitting.")

        train_ids, test_ids = train_test_split(
            ids, test_size=test_size, stratify=strata if not rare else None,
            random_state=self.random_state,
        )
        return set(train_ids), set(test_ids)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def build(self):
        """
        Full pipeline: load -> window -> featurize -> recording-level split.
        Returns train_X, train_y, test_X, test_y, meta (dict with window->
        recording provenance, for the RAG corpus partition step).
        """
        recordings = self.load_recordings()

        # NOTE: the exact schema of each record (which key holds KPI values,
        # which holds the anomaly condition type for stratification) must be
        # confirmed against the real downloaded data before this runs end to
        # end. Placeholder field names below — update after first real load.
        recording_labels = {}  # {recording_id: dominant_condition_type}
        all_windows = []  # list of (recording_id, start_idx, feature_vec, label)

        for rec_id, records in recordings.items():
            # placeholder extraction logic — adjust to real schema
            kpi_array = np.array([[r.get(ch, 0.0) for ch in NUMERICAL_CHANNELS] for r in records])
            ul_proto = records[0].get("UL_Protocol", "None")
            dl_proto = records[0].get("DL_Protocol", "None")
            protocol_onehot = self.encode_protocol(ul_proto, dl_proto)

            condition_type = records[0].get("anomaly_type", "normal")
            recording_labels[rec_id] = condition_type

            for start, window in self.sliding_windows(kpi_array):
                feat = self.extract_features(window, protocol_onehot)
                all_windows.append((rec_id, start, feat, condition_type))

        train_recs, test_recs = self.recording_level_split(recording_labels)

        train_X, train_y, test_X, test_y = [], [], [], []
        window_provenance = []  # for RAG corpus partitioning downstream

        for rec_id, start, feat, label in all_windows:
            window_provenance.append({"recording_id": rec_id, "start": start,
                                       "label": label, "split": "train" if rec_id in train_recs else "test"})
            if rec_id in train_recs:
                train_X.append(feat); train_y.append(label)
            else:
                test_X.append(feat); test_y.append(label)

        meta = {
            "n_recordings": len(recordings),
            "n_train_recordings": len(train_recs),
            "n_test_recordings": len(test_recs),
            "n_train_windows": len(train_X),
            "n_test_windows": len(test_X),
            "window_provenance": window_provenance,
        }

        print(f"Split: {len(train_recs)} train recordings / {len(test_recs)} test recordings")
        print(f"Windows: {len(train_X)} train / {len(test_X)} test")

        return (np.array(train_X), np.array(train_y),
                np.array(test_X), np.array(test_y), meta)


if __name__ == "__main__":
    builder = DatasetBuilder(data_dir="../../data/telecomts_raw")
    train_X, train_y, test_X, test_y, meta = builder.build()
    np.save("../../data/train_X.npy", train_X)
    np.save("../../data/train_y.npy", train_y)
    np.save("../../data/test_X.npy", test_X)
    np.save("../../data/test_y.npy", test_y)
    with open("../../data/split_meta.json", "w") as f:
        json.dump({k: v for k, v in meta.items() if k != "window_provenance"}, f, indent=2)
    print("Saved train/test arrays and split metadata to data/.")
