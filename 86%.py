"""
improved_video_pipeline_safe_label_loss.py

Robust fix for: TypeError: SparseCategoricalCrossentropy.__init__() got an unexpected keyword argument 'label_smoothing'

Strategy:
- Try to instantiate SparseCategoricalCrossentropy(with label_smoothing).
- If that fails, fall back to SparseCategoricalCrossentropy(without label_smoothing).
- Keep sparse integer labels throughout (no one-hot conversion).
"""

import os
import glob
import gc
import random
from collections import defaultdict
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.layers import (Input, Bidirectional, LSTM, Dense, Dropout,
                                     BatchNormalization, GlobalAveragePooling1D, GaussianNoise)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
import math

# ---------------- USER CONFIG ----------------
NO_STR_VIDEOS = r"C:\Users\user\OneDrive\Documents\video_model\trimmed_new_no_eyestrain"
STR_VIDEOS = r"C:\Users\user\OneDrive\Documents\video_model\trimmed_new_eyestrain"
FEATURE_DIR = r"C:\Users\user\OneDrive\Documents\video_model\precomputed_feats"

FRAME_SIZE = (224, 224)
BACKBONE = "mobilenetv2"
BATCH_FRAMES = 32           # frames per backbone batch

SEQ_LEN = 16                # number of frames per sequence (timesteps)
STEP = 8                    # used for building deterministic validation/test sequences
TEST_FRACTION = 0.15        # video-level test fraction
VAL_FRACTION_OF_TRAIN = 0.15
BATCH_SIZE = 16
EPOCHS = 50
RANDOM_SEED = 42
LR = 1e-4
L2_REG = 5e-4
DROPOUT = 0.5
MODEL_OUT = r"C:\Users\user\OneDrive\Documents\video_model\improved_video_model.keras"
SCALER_OUT = r"C:\Users\user\OneDrive\Documents\video_model\improved_scaler.save"
CONF_MATRIX_TEST = r"C:\Users\user\OneDrive\Documents\video_model\confusion_matrix_test.png"
CONF_MATRIX_ALL = r"C:\Users\user\OneDrive\Documents\video_model\confusion_matrix_all.png"
STEPS_PER_EPOCH = 200
# ----------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Enable GPU memory growth if GPUs present
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError:
        pass


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_video_files():
    vids = []
    labels = []
    for lbl, d in enumerate([NO_STR_VIDEOS, STR_VIDEOS]):  # 0=no strain, 1=strain
        if not os.path.isdir(d):
            print(f"Warning: directory not found: {d}")
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                vids.append(os.path.join(d, fn))
                labels.append(lbl)
    return vids, labels


def build_backbone(name="mobilenetv2"):
    if name.lower() == "mobilenetv2":
        base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg',
                           input_shape=(FRAME_SIZE[1], FRAME_SIZE[0], 3))
        preprocess = mobilenet_v2.preprocess_input
    else:
        raise NotImplementedError("Only MobileNetV2 supported in this script")
    return base, preprocess


def extract_frames_efficiently(video_path, max_frames=500):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        step = 1
    else:
        step = max(1, total_frames // max_frames)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, FRAME_SIZE)
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if len(frames) == 0:
        frames = [np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)]
    return np.array(frames, dtype=np.float32)


def precompute_all_features(video_paths, out_dir=FEATURE_DIR, backbone_name=BACKBONE):
    ensure_dir(out_dir)
    base, preprocess = build_backbone(backbone_name)
    feat_dim = base.output_shape[-1]
    print(f"[backbone] {backbone_name} -> feature dim {feat_dim}")

    for vp in video_paths:
        stem = os.path.splitext(os.path.basename(vp))[0]
        out_path = os.path.join(out_dir, f"{stem}.npy")
        if os.path.exists(out_path):
            continue
        try:
            frames = extract_frames_efficiently(vp)
            feats_list = []
            for i in range(0, frames.shape[0], BATCH_FRAMES):
                batch = frames[i:i + BATCH_FRAMES].astype('float32')
                batch = preprocess(batch)
                preds = base.predict(batch, verbose=0)
                feats_list.append(preds)
            feats = np.vstack(feats_list).astype(np.float32)
            np.save(out_path, feats)
            del frames, feats, feats_list
            gc.collect()
        except Exception as e:
            print(f"Error processing {vp}: {e}")
            continue
    print("Feature precompute complete.")


def build_deterministic_sequences(feature_dir=FEATURE_DIR):
    feature_files = sorted(glob.glob(os.path.join(feature_dir, "*.npy")))
    X_sequences = []
    y_labels = []
    video_groups = []
    video_to_sequences = defaultdict(list)
    video_label_map = {}
    for fpath in feature_files:
        stem = os.path.splitext(os.path.basename(fpath))[0]
        label = None
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            if os.path.exists(os.path.join(STR_VIDEOS, stem + ext)):
                label = 1
                break
            if os.path.exists(os.path.join(NO_STR_VIDEOS, stem + ext)):
                label = 0
                break
        if label is None:
            low = stem.lower()
            label = 1 if ("strain" in low and "no" not in low) else 0
        video_label_map[stem] = label

        features = np.load(fpath)
        n_frames = features.shape[0]
        if n_frames >= SEQ_LEN:
            for i in range(0, n_frames - SEQ_LEN + 1, STEP):
                seq = features[i:i + SEQ_LEN]
                X_sequences.append(seq)
                y_labels.append(label)
                video_groups.append(stem)
                video_to_sequences[stem].append(len(X_sequences) - 1)
        else:
            pad_count = SEQ_LEN - n_frames
            pad = np.repeat(features[-1:, :], pad_count, axis=0)
            seq = np.vstack([features, pad])
            X_sequences.append(seq)
            y_labels.append(label)
            video_groups.append(stem)
            video_to_sequences[stem].append(len(X_sequences) - 1)
    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    print(f"Built deterministic sequences: {len(X)} sequences from {len(video_label_map)} videos")
    return X, y, video_groups, video_to_sequences, video_label_map


class TrainSequenceGenerator(Sequence):
    """
    Samples sequences on the fly from per-video feature .npy arrays.
    Balanced sampling between classes inside a batch.
    RETURNS sparse integer labels (not one-hot).
    """

    def __init__(self, train_video_list, video_label_map, feature_dir, seq_len, batch_size, scaler,
                 steps_per_epoch=200, augment=True):
        self.train_videos = train_video_list
        self.video_label_map = video_label_map
        self.feature_dir = feature_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.augment = augment
        self.scaler = scaler
        self.class_videos = defaultdict(list)
        for v in self.train_videos:
            lbl = int(video_label_map[v])
            self.class_videos[lbl].append(v)
        self._feat_cache = {}
        self._cache_limit = 200

    def __len__(self):
        return int(self.steps_per_epoch)

    def __getitem__(self, idx):
        half = self.batch_size // 2
        stems = []
        labels = []
        for cls, n in [(0, half), (1, self.batch_size - half)]:
            pool = self.class_videos.get(cls, [])
            if len(pool) == 0:
                continue
            picks = random.choices(pool, k=n)
            stems.extend(picks)
            labels.extend([cls] * n)

        X_batch = np.zeros((len(stems), self.seq_len, self._feat_dim()), dtype=np.float32)
        for i, stem in enumerate(stems):
            feats = self._load_features(stem)
            n_frames = feats.shape[0]
            if n_frames <= self.seq_len:
                pad_count = self.seq_len - n_frames
                seq = np.vstack([feats, np.repeat(feats[-1:, :], pad_count, axis=0)])
            else:
                start = random.randint(0, n_frames - self.seq_len)
                seq = feats[start:start + self.seq_len].copy()
                if self.augment:
                    if random.random() < 0.15:
                        drop_idx = random.randint(0, self.seq_len - 1)
                        rep_from = min(self.seq_len - 1, max(0, drop_idx + random.choice([-1, 1])))
                        seq[drop_idx] = seq[rep_from]
                    if random.random() < 0.4:
                        seq += np.random.normal(scale=1e-3, size=seq.shape).astype(np.float32)
            X_batch[i] = seq

        # scaling
        resh = X_batch.reshape(-1, X_batch.shape[-1])
        resh = self.scaler.transform(resh)
        X_batch = resh.reshape(X_batch.shape)

        y_batch = np.array(labels, dtype=np.int32)  # sparse integer labels
        return X_batch, y_batch

    def on_epoch_end(self):
        if len(self._feat_cache) > self._cache_limit:
            self._feat_cache.clear()

    def _load_features(self, stem):
        if stem in self._feat_cache:
            return self._feat_cache[stem]
        path = os.path.join(self.feature_dir, f"{stem}.npy")
        feats = np.load(path).astype(np.float32)
        if len(self._feat_cache) < self._cache_limit:
            self._feat_cache[stem] = feats
        return feats

    def _feat_dim(self):
        if len(self._feat_cache) > 0:
            return next(iter(self._feat_cache.values())).shape[1]
        if len(self.train_videos) == 0:
            raise RuntimeError("No training videos provided to generator")
        v0 = self.train_videos[0]
        arr = np.load(os.path.join(self.feature_dir, f"{v0}.npy"))
        self._feat_cache[v0] = arr
        return arr.shape[1]


def build_regularized_lstm(input_shape, lstm_units=32):
    inp = Input(shape=input_shape)
    x = GaussianNoise(0.03)(inp)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True,
                           kernel_regularizer=l2(L2_REG),
                           recurrent_regularizer=l2(L2_REG),
                           dropout=0.25, recurrent_dropout=0.15))(x)
    x = GlobalAveragePooling1D()(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    out = Dense(2, activation='softmax', kernel_regularizer=l2(L2_REG))(x)

    model = Model(inp, out)

    # *** Robust loss creation: try label_smoothing; fallback if not supported
    try:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, label_smoothing=0.05)
        print("Using SparseCategoricalCrossentropy with label_smoothing=0.05")
    except TypeError:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        print("label_smoothing not supported for SparseCategoricalCrossentropy on this TF build — using without label_smoothing")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss=loss,
                  metrics=['accuracy'])
    return model


def predict_video_level(model, scaler, X_data, groups_list, video_to_sequences_map, video_label_map, batch_size=BATCH_SIZE):
    feat_dim = X_data.shape[2]
    X_scaled = np.zeros_like(X_data)
    chunk = 500
    for i in range(0, len(X_data), chunk):
        end = min(i + chunk, len(X_data))
        resh = X_data[i:end].reshape(-1, feat_dim)
        X_scaled[i:end] = scaler.transform(resh).reshape(X_data[i:end].shape)

    seq_probs = []
    for i in range(0, len(X_scaled), batch_size):
        end = min(i + batch_size, len(X_scaled))
        preds = model.predict(X_scaled[i:end], verbose=0)
        seq_probs.append(preds)
    seq_probs = np.vstack(seq_probs)

    video_stems = []
    video_preds = []
    video_probs = []
    video_true = []

    for stem, seq_indices in video_to_sequences_map.items():
        if len(seq_indices) == 0:
            continue
        seqs = [idx for idx in seq_indices if idx < seq_probs.shape[0]]
        if len(seqs) == 0:
            continue
        probs = seq_probs[seqs]
        avg = probs.mean(axis=0)
        pred = int(np.argmax(avg))
        video_stems.append(stem)
        video_preds.append(pred)
        video_probs.append(float(avg[1]))
        video_true.append(int(video_label_map.get(stem, 0)))
    return video_stems, np.array(video_preds), np.array(video_probs), np.array(video_true)


def plot_and_save_confusion(y_true, y_pred, out_path, title="Confusion Matrix"):
    if len(y_true) == 0:
        print("No samples to plot confusion matrix.")
        return
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 1:
        cm = np.array([[cm]])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    print("\n" + "=" * 60)
    print(f"{title}")
    print(f"Total samples: {len(y_true)}")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (strain class=1): {prec:.4f}")
    print(f"Recall (strain class=1): {recall:.4f}")
    print(f"F1 (strain class=1): {f1:.4f}")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Strain', 'Strain'], yticklabels=['No Strain', 'Strain'])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{title}\nTP={tp} FP={fp} FN={fn} TN={tn}")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100 if total > 0 else 0
            ax.text(j + 0.5, i + 0.3, f"{pct:.1f}%", ha='center', color='red', fontsize=10, weight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("\nClassification report (per-class):")
    print(classification_report(y_true, y_pred, target_names=["No Strain", "Strain"], digits=4))


def main():
    print("Starting improved pipeline (robust loss fix)...")
    ensure_dir(FEATURE_DIR)

    vids, vid_labels = find_video_files()
    print(f"Found {len(vids)} videos. Label dist: {np.bincount(vid_labels) if len(vid_labels)>0 else 'N/A'}")
    if len(vids) == 0:
        raise RuntimeError("No videos found. Check paths.")

    feature_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.npy")))
    if len(feature_files) < len(vids):
        print("Precomputing features (this may take a while)...")
        precompute_all_features(vids, out_dir=FEATURE_DIR, backbone_name=BACKBONE)
        feature_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.npy")))
    else:
        print(f"Using {len(feature_files)} existing .npy feature files")

    X_all, y_all, groups_all, video_to_seq_all, video_label_map = build_deterministic_sequences(FEATURE_DIR)
    if X_all.shape[0] == 0:
        raise RuntimeError("No sequences created - check feature files")

    unique_videos = list(video_to_seq_all.keys())
    video_labels = [video_label_map[v] for v in unique_videos]
    print(f"Unique videos: {len(unique_videos)} Label dist: {np.bincount(video_labels)}")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(gss.split(np.arange(len(unique_videos)), video_labels, groups=unique_videos))
    train_val_videos = [unique_videos[i] for i in train_val_idx]
    test_videos = [unique_videos[i] for i in test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION_OF_TRAIN, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss2.split(np.arange(len(train_val_videos)), [video_label_map[v] for v in train_val_videos], groups=train_val_videos))
    train_videos = [train_val_videos[i] for i in train_idx]
    val_videos = [train_val_videos[i] for i in val_idx]

    print(f"Videos -> Train: {len(train_videos)}, Val: {len(val_videos)}, Test: {len(test_videos)}")

    train_seq_idx = [i for i, g in enumerate(groups_all) if g in train_videos]
    val_seq_idx = [i for i, g in enumerate(groups_all) if g in val_videos]
    test_seq_idx = [i for i, g in enumerate(groups_all) if g in test_videos]

    X_train = X_all[train_seq_idx]; y_train = y_all[train_seq_idx]
    X_val = X_all[val_seq_idx]; y_val = y_all[val_seq_idx]
    X_test = X_all[test_seq_idx]; y_test = y_all[test_seq_idx]

    print(f"Sequences - Train: {len(X_train)} Val: {len(X_val)} Test: {len(X_test)}")

    feat_dim = X_train.shape[2]
    sample_size = min(3000, X_train.shape[0])
    sample_idx = np.random.choice(X_train.shape[0], sample_size, replace=False) if X_train.shape[0] > 0 else np.arange(0)
    sample_data = X_train[sample_idx].reshape(-1, feat_dim)
    scaler = StandardScaler()
    scaler.fit(sample_data)

    def scale_chunked(X_data, chunk_size=500):
        Xs = np.zeros_like(X_data)
        for i in range(0, len(X_data), chunk_size):
            end = min(i + chunk_size, len(X_data))
            resh = X_data[i:end].reshape(-1, feat_dim)
            Xs[i:end] = scaler.transform(resh).reshape(X_data[i:end].shape)
        return Xs

    X_val_s = scale_chunked(X_val)
    X_test_s = scale_chunked(X_test)
    X_all_s = scale_chunked(X_all)  # for all-video evaluation later

    train_gen = TrainSequenceGenerator(train_videos, video_label_map, FEATURE_DIR, SEQ_LEN, BATCH_SIZE,
                                       scaler, steps_per_epoch=STEPS_PER_EPOCH, augment=True)

    classes = np.unique(y_train)
    class_weights = {}
    if len(classes) > 0:
        class_weights_arr = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
    print("Class weights (for record):", class_weights)

    model = build_regularized_lstm((SEQ_LEN, feat_dim), lstm_units=32)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # fit uses sparse integer labels for validation
    history = model.fit(
        train_gen,
        validation_data=(X_val_s, y_val),
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=callbacks,
        class_weight=class_weights if class_weights else None,
        verbose=1
    )

    # Build test mapping (indices relative to X_test subset)
    test_groups_filtered = [groups_all[i] for i in test_seq_idx]
    test_video_to_seqs = defaultdict(list)
    for i, g in enumerate(test_groups_filtered):
        test_video_to_seqs[g].append(i)

    test_video_stems, test_video_preds, test_video_probs, test_video_true = predict_video_level(
        model, scaler, X_test, test_groups_filtered, test_video_to_seqs, video_label_map, batch_size=BATCH_SIZE
    )

    plot_and_save_confusion(test_video_true, test_video_preds, CONF_MATRIX_TEST, title="Test Set Video-Level Confusion Matrix")

    prec, rec, f1, _ = precision_recall_fscore_support(test_video_true, test_video_preds, average='binary', pos_label=1, zero_division=0)
    acc = accuracy_score(test_video_true, test_video_preds)
    print("\nTest-set metrics:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    all_video_stems, all_video_preds, all_video_probs, all_video_true = predict_video_level(
        model, scaler, X_all, groups_all, video_to_seq_all, video_label_map, batch_size=BATCH_SIZE
    )
    plot_and_save_confusion(all_video_true, all_video_preds, CONF_MATRIX_ALL, title="All Videos Video-Level Confusion Matrix")
    prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(all_video_true, all_video_preds, average='binary', pos_label=1, zero_division=0)
    acc_a = accuracy_score(all_video_true, all_video_preds)
    print("\nALL-videos metrics:")
    print(f"Accuracy: {acc_a:.4f}, Precision: {prec_a:.4f}, Recall: {rec_a:.4f}, F1: {f1_a:.4f}")

    model.save(MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print(f"Saved model to: {MODEL_OUT}")
    print(f"Saved scaler to: {SCALER_OUT}")

    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'test_video_info': (test_video_stems, test_video_preds, test_video_probs, test_video_true),
        'all_video_info': (all_video_stems, all_video_preds, all_video_probs, all_video_true)
    }


if __name__ == "__main__":
    results = main()
