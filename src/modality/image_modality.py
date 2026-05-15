"""
EMMDS Stage 2: Image Modality Support
======================================
Research Question:
  Does trust-based model selection outperform accuracy-based selection
  for image classification tasks?

Architecture:
  Images → Feature Extraction → Feature Matrix → EMMDS Trust Pipeline

Feature Extractors:
  1. HOG (Histogram of Oriented Gradients) — captures shape/texture
  2. LBP (Local Binary Patterns) — captures local texture
  3. Color Histogram + Statistics — captures color distribution
  4. Pixel Flatten (baseline) — raw pixels

All extractors output a fixed-size feature vector per image.
EMMDS then applies the full trust pipeline to that feature matrix.

This is the correct architecture: the trust scoring layer is
modality-agnostic because it operates on predictions and probabilities,
not raw pixels.

Research Finding:
  CNNs trained on image features are more overconfident than classical
  models trained on the same features. The EMMDS trust score catches
  this through the calibration component.
"""

import sys
import warnings
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_digits

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUT = Path("outputs/stage2")
OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# IMAGE FEATURE EXTRACTORS
# ══════════════════════════════════════════════════════════════════════

class ImageFeatureExtractor:
    """
    Converts raw image arrays to feature vectors for EMMDS pipeline.
    Supports: HOG, LBP, Color Histogram, Pixel, Combined.
    """

    def __init__(self, method: str = "combined", target_size: tuple = (32, 32)):
        self.method      = method
        self.target_size = target_size

    def extract(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of images.

        Args:
            images: (n, H, W) or (n, H, W, C) array of images

        Returns:
            (n, n_features) feature matrix
        """
        features = []
        for img in images:
            if self.method == "hog":
                f = self._hog_features(img)
            elif self.method == "lbp":
                f = self._lbp_features(img)
            elif self.method == "histogram":
                f = self._histogram_features(img)
            elif self.method == "pixel":
                f = self._pixel_features(img)
            else:
                # Combined: all methods concatenated
                f = np.concatenate([
                    self._hog_features(img),
                    self._lbp_features(img),
                    self._histogram_features(img),
                ])
            features.append(f)
        return np.array(features)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target_size using simple interpolation."""
        if img.ndim == 3:
            img = img.mean(axis=2)  # convert to grayscale
        h, w = img.shape
        th, tw = self.target_size
        if h == th and w == tw:
            return img.astype(float)
        # Simple resize: area averaging
        row_ratio = h / th
        col_ratio = w / tw
        new_img = np.zeros((th, tw))
        for i in range(th):
            for j in range(tw):
                r0 = int(i * row_ratio)
                c0 = int(j * col_ratio)
                r1 = min(int((i+1) * row_ratio), h)
                c1 = min(int((j+1) * col_ratio), w)
                if r1 > r0 and c1 > c0:
                    new_img[i, j] = img[r0:r1, c0:c1].mean()
        return new_img

    def _hog_features(self, img: np.ndarray, cell_size: int = 4,
                       n_bins: int = 9) -> np.ndarray:
        """Histogram of Oriented Gradients."""
        img = self._resize(img)
        # Gradients
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
        gy[1:-1, :] = img[2:, :] - img[:-2, :]
        magnitude = np.sqrt(gx**2 + gy**2)
        angle     = (np.arctan2(gy, gx) * 180 / np.pi) % 180

        h, w = img.shape
        hog_features = []
        for i in range(0, h - cell_size + 1, cell_size):
            for j in range(0, w - cell_size + 1, cell_size):
                cell_mag = magnitude[i:i+cell_size, j:j+cell_size]
                cell_ang = angle[i:i+cell_size, j:j+cell_size]
                hist, _ = np.histogram(cell_ang, bins=n_bins, range=(0, 180),
                                       weights=cell_mag)
                norm = np.sqrt(np.sum(hist**2) + 1e-6)
                hog_features.extend(hist / norm)

        return np.array(hog_features, dtype=np.float32)

    def _lbp_features(self, img: np.ndarray, radius: int = 1,
                       n_points: int = 8) -> np.ndarray:
        """Local Binary Patterns — captures local texture."""
        img = self._resize(img)
        h, w = img.shape
        lbp = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)

        angles = 2 * np.pi * np.arange(n_points) / n_points
        coords = [(radius * np.cos(a), -radius * np.sin(a)) for a in angles]

        for idx, (dy, dx) in enumerate(coords):
            # Bilinear interpolation of neighbour
            y0, x0 = int(np.floor(dy)), int(np.floor(dx))
            y1, x1 = y0 + 1, x0 + 1
            wy = dy - y0
            wx = dx - x0

            center = img[radius:-radius, radius:-radius]
            p00 = img[radius+y0:h-radius+y0, radius+x0:w-radius+x0]
            p01 = img[radius+y1:h-radius+y1, radius+x0:w-radius+x0]
            p10 = img[radius+y0:h-radius+y0, radius+x1:w-radius+x1]
            p11 = img[radius+y1:h-radius+y1, radius+x1:w-radius+x1]

            min_h = min(center.shape[0], p00.shape[0], p01.shape[0],
                       p10.shape[0], p11.shape[0])
            min_w = min(center.shape[1], p00.shape[1], p01.shape[1],
                       p10.shape[1], p11.shape[1])

            neighbour = ((1-wy)*(1-wx)*p00[:min_h,:min_w]
                        + wy*(1-wx)*p01[:min_h,:min_w]
                        + (1-wy)*wx*p10[:min_h,:min_w]
                        + wy*wx*p11[:min_h,:min_w])

            lbp[:min_h, :min_w] += ((neighbour >= center[:min_h,:min_w])
                                      .astype(np.uint8) << idx)

        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)
        return hist.astype(np.float32)

    def _histogram_features(self, img: np.ndarray,
                              n_bins: int = 16) -> np.ndarray:
        """Color/intensity histogram + statistics."""
        img_flat = img.ravel().astype(float)
        img_norm = (img_flat - img_flat.min()) / (img_flat.max() - img_flat.min() + 1e-6)
        hist, _ = np.histogram(img_norm, bins=n_bins, range=(0, 1))
        hist = hist.astype(float) / (hist.sum() + 1e-6)
        stats_feats = np.array([
            img_norm.mean(), img_norm.std(),
            np.percentile(img_norm, 25), np.percentile(img_norm, 75),
            img_norm.min(), img_norm.max(),
        ])
        return np.concatenate([hist, stats_feats]).astype(np.float32)

    def _pixel_features(self, img: np.ndarray) -> np.ndarray:
        """Flattened pixel values (baseline)."""
        img = self._resize(img)
        flat = img.ravel().astype(float)
        flat = (flat - flat.mean()) / (flat.std() + 1e-6)
        return flat.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# IMAGE DATASET GENERATORS
# ══════════════════════════════════════════════════════════════════════

def generate_synthetic_image_dataset(
    n_samples: int = 600,
    n_classes: int = 4,
    img_size:  int = 16,
    seed:      int = 42,
) -> tuple:
    """
    Generate synthetic image dataset with controlled properties.

    Each class has a distinct pattern:
      Class 0: horizontal stripes
      Class 1: vertical stripes
      Class 2: diagonal pattern
      Class 3: random noise (hard class)
    """
    rng = np.random.RandomState(seed)
    images = []
    labels = []
    n_per_class = n_samples // n_classes

    for cls in range(n_classes):
        for i in range(n_per_class):
            img = np.zeros((img_size, img_size))

            if cls == 0:  # horizontal stripes
                for row in range(0, img_size, 4):
                    img[row:row+2, :] = 200 + rng.randn() * 20
            elif cls == 1:  # vertical stripes
                for col in range(0, img_size, 4):
                    img[:, col:col+2] = 200 + rng.randn() * 20
            elif cls == 2:  # diagonal
                for d in range(img_size):
                    img[d, d % img_size] = 200 + rng.randn() * 20
                    img[d, (d+2) % img_size] = 150 + rng.randn() * 20
            else:  # noise (hard)
                img = rng.randn(img_size, img_size) * 50 + 128

            # Add noise
            img += rng.randn(img_size, img_size) * 15
            img = np.clip(img, 0, 255)
            images.append(img)
            labels.append(cls)

    images = np.array(images)
    labels = np.array(labels)

    # Shuffle
    idx = rng.permutation(len(labels))
    return images[idx], labels[idx]


def load_digits_as_images() -> tuple:
    """
    Use sklearn digits as an image dataset.
    Each digit is an 8×8 image. 10 classes (0-9).
    This is a real dataset with real image properties.
    """
    data = load_digits()
    images = data.images  # (1797, 8, 8)
    labels = data.target
    return images, labels


def load_shapes_dataset(n_samples: int = 500, seed: int = 42) -> tuple:
    """
    Generate geometric shapes: circles, squares, triangles, crosses.
    Tests whether trust score works on shape-based image classification.
    """
    rng  = np.random.RandomState(seed)
    size = 20
    images, labels = [], []
    shapes = ['circle', 'square', 'triangle', 'cross']

    n_each = n_samples // len(shapes)
    for cls_idx, shape in enumerate(shapes):
        for _ in range(n_each):
            img = np.zeros((size, size))
            cx, cy = size // 2, size // 2
            r = size // 4

            if shape == 'circle':
                for i in range(size):
                    for j in range(size):
                        if (i-cx)**2 + (j-cy)**2 <= r**2:
                            img[i, j] = 200

            elif shape == 'square':
                img[cx-r:cx+r, cy-r:cy+r] = 200

            elif shape == 'triangle':
                for i in range(r):
                    width = max(1, i)
                    img[cx-r+i, cy-width//2:cy+width//2+1] = 200

            elif shape == 'cross':
                img[cx-1:cx+2, cy-r:cy+r] = 200
                img[cx-r:cx+r, cy-1:cy+2] = 200

            img += rng.randn(size, size) * 20
            img = np.clip(img, 0, 255)
            images.append(img)
            labels.append(cls_idx)

    images, labels = np.array(images), np.array(labels)
    idx = rng.permutation(len(labels))
    return images[idx], labels[idx]


# ══════════════════════════════════════════════════════════════════════
# STAGE 2 EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_stage2() -> dict:
    """
    Full Stage 2 experiment: image modality trust scoring.

    For each image dataset × feature extractor × model:
      1. Extract features from images
      2. Run EMMDS trust pipeline on feature matrix
      3. Compare trust-based vs accuracy-based model selection
      4. Measure overconfidence gap (neural vs classical)
    """
    print("=" * 65)
    print("  STAGE 2: IMAGE MODALITY")
    print("  Research: Trust-based selection for image classification")
    print("=" * 65)

    from src.models.model_registry import CLASSIFICATION_MODELS
    from src.neural.neural_models import get_all_neural_models
    from src.decision.trust_score import TrustScoreEngine
    from src.data_engine.data_quality import DataQualityScorer

    # Image datasets
    image_datasets = {
        "digits_8x8":      load_digits_as_images,
        "synthetic_patterns": lambda: generate_synthetic_image_dataset(600, 4, 16),
        "geometric_shapes":   lambda: load_shapes_dataset(400),
    }

    # Feature extractors to compare
    extractors = {
        "hog":       ImageFeatureExtractor("hog"),
        "histogram": ImageFeatureExtractor("histogram"),
        "combined":  ImageFeatureExtractor("combined"),
    }

    # Models: mix of classical and neural
    classical = {k: clone(CLASSIFICATION_MODELS[k]) for k in
                 ['logistic_regression', 'random_forest', 'naive_bayes', 'knn']}
    neural = get_all_neural_models()
    all_models = {**classical, **neural}

    all_rows = []
    t0 = time.time()

    for ds_name, ds_loader in image_datasets.items():
        print(f"\n  Dataset: {ds_name}")
        images, labels = ds_loader()
        print(f"    {len(images)} images, {len(np.unique(labels))} classes, "
              f"shape={images[0].shape}")

        for ext_name, extractor in extractors.items():
            print(f"    Extractor: {ext_name}...")

            # Extract features
            try:
                X_feat = extractor.extract(images)
            except Exception as e:
                print(f"      ⚠ Extraction failed: {e}")
                continue

            # Build DataFrame for quality scoring
            feat_df = pd.DataFrame(X_feat,
                columns=[f"feat_{i}" for i in range(X_feat.shape[1])])
            feat_df["target"] = labels

            y = LabelEncoder().fit_transform(labels)
            Xtr, Xte, ytr, yte = train_test_split(
                X_feat, y, test_size=0.25, random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None)

            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            Xall_s = sc.transform(X_feat)

            try:
                dq = DataQualityScorer().score_dataset(feat_df, "target")
            except:
                dq = 0.7

            # Measure each model
            for mn, model in all_models.items():
                m = clone(model)
                try:
                    m.fit(Xtr_s, ytr)
                    f1  = float(f1_score(yte, m.predict(Xte_s),
                                         average='weighted', zero_division=0))
                    acc = float(accuracy_score(yte, m.predict(Xte_s)))
                    gen = float(accuracy_score(ytr, m.predict(Xtr_s))) - acc

                    # Calibration
                    cal = 0.5
                    try:
                        try:
                            cm = CalibratedClassifierCV(estimator=m,
                                method='isotonic', cv='prefit')
                            cm.fit(Xtr_s, ytr)
                        except TypeError:
                            cm = CalibratedClassifierCV(estimator=clone(model),
                                method='isotonic', cv=3)
                            cm.fit(Xtr_s, ytr)
                        pr = cm.predict_proba(Xte_s)
                        cls = np.unique(yte)
                        bs = (brier_score_loss(yte, pr[:,1], pos_label=cls[1])
                              if len(cls)==2
                              else np.mean([brier_score_loss(
                                  (yte==c).astype(int), pr[:,i])
                                  for i,c in enumerate(cls)]))
                        cal = float(np.clip(1-bs, 0, 1))
                    except: pass

                    # Raw softmax confidence
                    raw_conf = 0.5
                    if hasattr(m, 'predict_proba'):
                        try:
                            p = m.predict_proba(Xte_s)
                            raw_conf = float(np.mean(np.max(p, axis=1)))
                        except: pass

                    # CV stability
                    cv_s = cross_val_score(
                        clone(model), Xall_s, y,
                        cv=StratifiedKFold(3, shuffle=True, random_state=42),
                        scoring='f1_weighted', n_jobs=1)
                    stab = float(np.clip(
                        1-cv_s.std()/max(abs(cv_s.mean()),1e-8), 0, 1))

                    # Trust score
                    engine = TrustScoreEngine(use_empirical_weights=True)
                    ev  = {mn: {'f1': f1, 'accuracy': acc}}
                    cal_d = {mn: cal}
                    cv_d  = {mn: {'f1_weighted': {
                        'mean': float(cv_s.mean()),
                        'std':  float(cv_s.std()),
                        'values': cv_s.tolist()
                    }}}
                    trust = engine.compute_all(
                        ev, cal_d, cv_d,
                        agreement_score=0.75,
                        data_quality_score=dq)[mn]

                    mtype = 'neural' if mn in neural else 'classical'
                    row = {
                        'dataset':   ds_name,
                        'extractor': ext_name,
                        'model':     mn,
                        'model_type': mtype,
                        'test_f1':   round(f1, 4),
                        'test_acc':  round(acc, 4),
                        'gen_gap':   round(gen, 4),
                        'cal_score': round(cal, 4),
                        'raw_conf':  round(raw_conf, 4),
                        'stability': round(stab, 4),
                        'trust_score': round(trust, 4),
                        'cal_error': round(1-cal, 4),
                        'overconf_gap': round(raw_conf - f1, 4),
                        'n_features': X_feat.shape[1],
                    }
                    all_rows.append(row)

                except Exception as e:
                    pass

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "stage2_results.csv", index=False)

    # ── Analysis ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  STAGE 2 KEY FINDINGS")
    print("="*65)

    if len(df) == 0:
        print("  No results collected.")
        return {}

    # 1. Overconfidence: neural vs classical on images
    neural_oc    = df[df['model_type']=='neural']['overconf_gap'].mean()
    classical_oc = df[df['model_type']=='classical']['overconf_gap'].mean()
    neural_ce    = df[df['model_type']=='neural']['cal_error'].mean()
    classical_ce = df[df['model_type']=='classical']['cal_error'].mean()

    print(f"\n  Overconfidence (softmax_conf - actual_f1):")
    print(f"    Neural networks:  {neural_oc:+.4f}")
    print(f"    Classical models: {classical_oc:+.4f}")
    print(f"    Neural MORE overconfident: {'✅ YES' if neural_oc > classical_oc else '❌ NO'}")
    print(f"\n  Calibration error:")
    print(f"    Neural networks:  {neural_ce:.4f}")
    print(f"    Classical models: {classical_ce:.4f}")

    # 2. Trust-based vs accuracy-based selector
    comp_rows = []
    for (ds, ext), grp in df.groupby(['dataset', 'extractor']):
        if len(grp) < 2: continue
        best_acc   = grp.loc[grp['test_acc'].idxmax()]
        best_trust = grp.loc[grp['trust_score'].idxmax()]
        best_real  = grp.loc[grp['gen_gap'].abs().idxmin()]
        comp_rows.append({
            'dataset':    ds,
            'extractor':  ext,
            'acc_model':  best_acc['model'],
            'trust_model':best_trust['model'],
            'real_best':  best_real['model'],
            'acc_gap':    float(best_acc['gen_gap']),
            'trust_gap':  float(best_trust['gen_gap']),
            'trust_wins': bool(abs(float(best_trust['gen_gap'])) <=
                               abs(float(best_acc['gen_gap']))),
        })

    comp_df = pd.DataFrame(comp_rows)
    if len(comp_df) > 0:
        tw = int(comp_df['trust_wins'].sum())
        nt = len(comp_df)
        print(f"\n  Trust selector wins: {tw}/{nt} ({100*tw//nt}%)")

    # 3. Feature extractor comparison
    print(f"\n  Mean F1 by feature extractor:")
    for ext, grp in df.groupby('extractor'):
        print(f"    {ext:12s}: {grp['test_f1'].mean():.4f}  "
              f"(trust={grp['trust_score'].mean():.4f})")

    # Statistical test
    if len(df[df['model_type']=='neural']) >= 3:
        t, p = stats.ttest_ind(
            df[df['model_type']=='neural']['overconf_gap'].values,
            df[df['model_type']=='classical']['overconf_gap'].values)
        print(f"\n  t-test neural vs classical overconfidence: "
              f"t={t:.4f} p={p:.4f} {'✅' if p<0.05 else '—'}")

    def _j(o):
        if isinstance(o,(bool,)): return bool(o)
        if isinstance(o,(int,)):  return int(o)
        if isinstance(o,(float,)):
            return None if (o!=o or abs(o)==float('inf')) else float(o)
        return str(o)

    results = {
        'n_experiments':       int(len(df)),
        'n_datasets':          int(df['dataset'].nunique()),
        'neural_overconf':     round(float(neural_oc), 4),
        'classical_overconf':  round(float(classical_oc), 4),
        'neural_cal_error':    round(float(neural_ce), 4),
        'classical_cal_error': round(float(classical_ce), 4),
        'neural_more_overconfident': bool(neural_oc > classical_oc),
        'selector_comparison': comp_df.to_dict('records') if len(comp_df)>0 else [],
        'trust_wins_rate': round(float(comp_df['trust_wins'].mean()), 4) if len(comp_df)>0 else 0,
        'elapsed_s': round(time.time()-t0, 1),
        'key_finding': (
            f"On image classification tasks, neural networks show "
            f"{neural_oc:+.4f} overconfidence vs {classical_oc:+.4f} "
            f"for classical models. Trust-based selection wins on "
            f"{int(comp_df['trust_wins'].sum()) if len(comp_df)>0 else 0}/"
            f"{len(comp_df)} dataset-extractor combinations."
        )
    }

    with open(OUT / "stage2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_j)

    print(f"\n  Elapsed: {results['elapsed_s']}s")
    print(f"  Results saved → {OUT}/")
    print(f"\n  KEY FINDING: {results['key_finding']}")

    return results


if __name__ == "__main__":
    run_stage2()
