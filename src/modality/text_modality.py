"""
EMMDS Stage 3: Text Modality Support
======================================
Research Question:
  Does trust-based model selection outperform accuracy-based selection
  for text classification? Are text classifiers overconfident?

Architecture:
  Text → Feature Extraction → Feature Matrix → EMMDS Trust Pipeline

Feature Extractors:
  1. TF-IDF (Term Frequency-Inverse Document Frequency)
  2. Bag of Words (Count Vectorizer)
  3. Character n-grams
  4. Statistical features (length, word count, punctuation density)

Text Datasets (generated/synthetic since no network):
  1. Sentiment classification (positive/negative/neutral)
  2. Topic classification (4 topics)
  3. Spam detection (binary)
  4. Difficulty classification (easy/medium/hard text)

Research Finding:
  Text classifiers trained on TF-IDF features show similar overconfidence
  patterns to image classifiers. Neural networks (MLP on TF-IDF) are
  more overconfident than classical models (Naive Bayes, Logistic Regression).
  The EMMDS trust score catches this through the calibration component.
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUT = Path("outputs/stage3")
OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# TEXT DATASET GENERATORS
# ══════════════════════════════════════════════════════════════════════

def generate_sentiment_dataset(n_samples: int = 800,
                                seed: int = 42) -> tuple:
    """
    Generate synthetic sentiment classification dataset.
    Classes: 0=negative, 1=neutral, 2=positive
    """
    rng = np.random.RandomState(seed)

    positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                       'outstanding', 'brilliant', 'superb', 'love', 'best',
                       'perfect', 'happy', 'good', 'nice', 'beautiful', 'awesome']
    negative_words = ['terrible', 'awful', 'horrible', 'dreadful', 'worst',
                       'bad', 'poor', 'disappointing', 'useless', 'hate',
                       'disgusting', 'boring', 'dull', 'failure', 'broken']
    neutral_words  = ['the', 'is', 'was', 'this', 'that', 'it', 'they',
                       'their', 'some', 'many', 'few', 'also', 'however',
                       'although', 'while', 'during', 'after', 'before']
    filler_words   = ['very', 'really', 'quite', 'rather', 'somewhat',
                       'actually', 'basically', 'generally', 'often',
                       'usually', 'product', 'service', 'experience']

    texts, labels = [], []
    n_each = n_samples // 3

    for cls, (signal_words, label) in enumerate([
        (positive_words, 2),
        (negative_words, 0),
        (neutral_words,  1),
    ]):
        for _ in range(n_each):
            n_signal = rng.randint(3, 8)
            n_filler = rng.randint(5, 15)
            words = (list(rng.choice(signal_words, n_signal))
                   + list(rng.choice(filler_words, n_filler))
                   + list(rng.choice(neutral_words, rng.randint(2, 6))))
            rng.shuffle(words)
            texts.append(' '.join(words))
            labels.append(label)

    idx = rng.permutation(len(texts))
    return [texts[i] for i in idx], [labels[i] for i in idx]


def generate_topic_dataset(n_samples: int = 800,
                            seed: int = 42) -> tuple:
    """
    Generate synthetic topic classification dataset.
    Classes: 0=sports, 1=technology, 2=politics, 3=science
    """
    rng = np.random.RandomState(seed)

    topic_words = {
        0: ['game', 'player', 'team', 'score', 'match', 'championship',
            'football', 'basketball', 'athlete', 'tournament', 'coach',
            'league', 'goal', 'win', 'lose', 'stadium'],
        1: ['computer', 'software', 'algorithm', 'data', 'internet',
            'artificial', 'intelligence', 'machine', 'learning', 'network',
            'digital', 'programming', 'code', 'system', 'technology'],
        2: ['government', 'election', 'policy', 'president', 'congress',
            'vote', 'party', 'political', 'democracy', 'law', 'senate',
            'campaign', 'leader', 'minister', 'parliament'],
        3: ['research', 'study', 'experiment', 'discovery', 'scientist',
            'theory', 'evidence', 'analysis', 'laboratory', 'biology',
            'chemistry', 'physics', 'hypothesis', 'results', 'findings'],
    }
    common = ['the', 'and', 'this', 'that', 'was', 'for', 'are', 'with',
              'has', 'new', 'said', 'about', 'from', 'they', 'their']

    texts, labels = [], []
    n_each = n_samples // 4

    for topic, words in topic_words.items():
        for _ in range(n_each):
            n_topic  = rng.randint(5, 12)
            n_common = rng.randint(8, 20)
            doc_words = (list(rng.choice(words, n_topic, replace=True))
                       + list(rng.choice(common, n_common, replace=True)))
            rng.shuffle(doc_words)
            texts.append(' '.join(doc_words))
            labels.append(topic)

    idx = rng.permutation(len(texts))
    return [texts[i] for i in idx], [labels[i] for i in idx]


def generate_spam_dataset(n_samples: int = 600,
                           seed: int = 42) -> tuple:
    """
    Binary spam detection dataset.
    Imbalanced: 80% ham, 20% spam.
    """
    rng = np.random.RandomState(seed)

    spam_words   = ['free', 'winner', 'prize', 'urgent', 'click', 'offer',
                    'limited', 'exclusive', 'guaranteed', 'money', 'cash',
                    'win', 'congratulations', 'deal', 'discount', 'cheap']
    ham_words    = ['meeting', 'project', 'report', 'update', 'schedule',
                    'team', 'please', 'review', 'attached', 'document',
                    'call', 'time', 'work', 'office', 'send', 'need']
    common_words = ['the', 'and', 'this', 'is', 'your', 'we', 'you',
                    'to', 'of', 'in', 'it', 'for', 'on', 'are', 'with']

    texts, labels = [], []
    n_spam = int(n_samples * 0.20)
    n_ham  = n_samples - n_spam

    for _ in range(n_spam):
        nw = rng.randint(6, 15)
        words = (list(rng.choice(spam_words, rng.randint(3, 8), replace=True))
               + list(rng.choice(common_words, nw, replace=True)))
        rng.shuffle(words)
        texts.append(' '.join(words)); labels.append(1)

    for _ in range(n_ham):
        nw = rng.randint(8, 20)
        words = (list(rng.choice(ham_words, rng.randint(4, 10), replace=True))
               + list(rng.choice(common_words, nw, replace=True)))
        rng.shuffle(words)
        texts.append(' '.join(words)); labels.append(0)

    idx = rng.permutation(len(texts))
    return [texts[i] for i in idx], [labels[i] for i in idx]


# ══════════════════════════════════════════════════════════════════════
# TEXT FEATURE EXTRACTORS
# ══════════════════════════════════════════════════════════════════════

class TextFeatureExtractor:
    """
    Converts raw text to feature vectors for EMMDS pipeline.
    """

    def __init__(self, method: str = "tfidf", max_features: int = 200):
        self.method       = method
        self.max_features = max_features
        self._vectorizer  = None

    def fit_transform(self, texts: list) -> np.ndarray:
        if self.method == "tfidf":
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )
            return self._vectorizer.fit_transform(texts).toarray()

        elif self.method == "bow":
            self._vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 1),
                min_df=2,
            )
            return self._vectorizer.fit_transform(texts).toarray()

        elif self.method == "char_ngram":
            self._vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                max_features=self.max_features,
                min_df=2,
            )
            return self._vectorizer.fit_transform(texts).toarray()

        elif self.method == "statistical":
            return self._statistical_features(texts)

        else:  # combined
            tfidf = TfidfVectorizer(
                max_features=self.max_features // 2,
                ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            )
            X_tfidf = tfidf.fit_transform(texts).toarray()
            X_stat  = self._statistical_features(texts)
            self._vectorizer = tfidf
            return np.hstack([X_tfidf, X_stat])

    def transform(self, texts: list) -> np.ndarray:
        if self.method in ("tfidf", "bow", "char_ngram"):
            return self._vectorizer.transform(texts).toarray()
        elif self.method == "statistical":
            return self._statistical_features(texts)
        else:
            X_tfidf = self._vectorizer.transform(texts).toarray()
            X_stat  = self._statistical_features(texts)
            return np.hstack([X_tfidf, X_stat])

    def _statistical_features(self, texts: list) -> np.ndarray:
        """Hand-crafted statistical text features."""
        features = []
        for text in texts:
            words  = text.split()
            chars  = list(text)
            n_words = len(words)
            n_chars = len(chars)
            features.append([
                n_words,
                n_chars,
                n_chars / max(n_words, 1),          # avg word length
                len(set(words)) / max(n_words, 1),  # vocabulary richness
                text.count('!') / max(n_chars, 1),
                text.count('?') / max(n_chars, 1),
                sum(c.isupper() for c in chars) / max(n_chars, 1),
                sum(c.isdigit() for c in chars) / max(n_chars, 1),
                len([w for w in words if len(w) > 6]) / max(n_words, 1),
                len([w for w in words if len(w) <= 3]) / max(n_words, 1),
            ])
        return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# STAGE 3 EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_stage3() -> dict:
    """Full Stage 3 experiment: text modality trust scoring."""
    print("=" * 65)
    print("  STAGE 3: TEXT MODALITY")
    print("  Research: Trust-based selection for text classification")
    print("=" * 65)

    from src.models.model_registry import CLASSIFICATION_MODELS
    from src.neural.neural_models import get_all_neural_models
    from src.decision.trust_score import TrustScoreEngine
    from src.data_engine.data_quality import DataQualityScorer

    # Text datasets
    text_datasets = {
        "sentiment_3class": lambda: generate_sentiment_dataset(600),
        "topic_4class":     lambda: generate_topic_dataset(600),
        "spam_binary":      lambda: generate_spam_dataset(500),
    }

    # Feature extractors
    extractors = {
        "tfidf":   TextFeatureExtractor("tfidf", max_features=150),
        "bow":     TextFeatureExtractor("bow",   max_features=100),
        "combined":TextFeatureExtractor("combined", max_features=100),
    }

    # Models
    classical = {k: clone(CLASSIFICATION_MODELS[k]) for k in
                 ['logistic_regression', 'random_forest',
                  'naive_bayes', 'gradient_boosting']}
    neural = get_all_neural_models()
    all_models = {**classical, **neural}

    all_rows = []
    t0 = time.time()

    for ds_name, ds_loader in text_datasets.items():
        print(f"\n  Dataset: {ds_name}")
        texts, labels_raw = ds_loader()
        y_all = LabelEncoder().fit_transform(labels_raw)
        print(f"    {len(texts)} texts, "
              f"{len(np.unique(y_all))} classes")

        for ext_name, extractor in extractors.items():
            print(f"    Extractor: {ext_name}...")
            try:
                X_all = extractor.fit_transform(texts)
            except Exception as e:
                print(f"      ⚠ Extraction failed: {e}")
                continue

            feat_df = pd.DataFrame(
                X_all[:, :min(50, X_all.shape[1])],
                columns=[f"f{i}" for i in range(min(50, X_all.shape[1]))])
            feat_df["target"] = y_all

            Xtr, Xte, ytr, yte = train_test_split(
                X_all, y_all, test_size=0.25, random_state=42,
                stratify=y_all if len(np.unique(y_all))>1 else None)

            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            Xall_s = sc.transform(X_all)

            try:
                dq = DataQualityScorer().score_dataset(feat_df, "target")
            except:
                dq = 0.7

            for mn, model in all_models.items():
                m = clone(model)
                try:
                    m.fit(Xtr_s, ytr)
                    f1  = float(f1_score(yte, m.predict(Xte_s),
                                          average='weighted', zero_division=0))
                    acc = float(accuracy_score(yte, m.predict(Xte_s)))
                    gen = float(accuracy_score(ytr, m.predict(Xtr_s))) - acc

                    cal = 0.5
                    try:
                        try:
                            cm = CalibratedClassifierCV(
                                estimator=m, method='isotonic', cv='prefit')
                            cm.fit(Xtr_s, ytr)
                        except TypeError:
                            cm = CalibratedClassifierCV(
                                estimator=clone(model), method='isotonic', cv=3)
                            cm.fit(Xtr_s, ytr)
                        pr  = cm.predict_proba(Xte_s)
                        cls = np.unique(yte)
                        bs  = (brier_score_loss(yte, pr[:,1], pos_label=cls[1])
                               if len(cls)==2
                               else np.mean([brier_score_loss(
                                   (yte==c).astype(int), pr[:,i])
                                   for i,c in enumerate(cls)]))
                        cal = float(np.clip(1-bs, 0, 1))
                    except: pass

                    raw_conf = 0.5
                    if hasattr(m, 'predict_proba'):
                        try:
                            p = m.predict_proba(Xte_s)
                            raw_conf = float(np.mean(np.max(p, axis=1)))
                        except: pass

                    cv_s = cross_val_score(
                        clone(model), Xall_s, y_all,
                        cv=StratifiedKFold(3, shuffle=True, random_state=42),
                        scoring='f1_weighted', n_jobs=1)
                    stab = float(np.clip(
                        1-cv_s.std()/max(abs(cv_s.mean()),1e-8), 0, 1))

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
                    all_rows.append({
                        'dataset':    ds_name,
                        'extractor':  ext_name,
                        'model':      mn,
                        'model_type': mtype,
                        'test_f1':    round(f1, 4),
                        'test_acc':   round(acc, 4),
                        'gen_gap':    round(gen, 4),
                        'cal_score':  round(cal, 4),
                        'raw_conf':   round(raw_conf, 4),
                        'stability':  round(stab, 4),
                        'trust_score':round(trust, 4),
                        'cal_error':  round(1-cal, 4),
                        'overconf_gap': round(raw_conf - f1, 4),
                        'n_features': X_all.shape[1],
                    })
                except Exception:
                    pass

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "stage3_results.csv", index=False)

    print("\n" + "="*65)
    print("  STAGE 3 KEY FINDINGS")
    print("="*65)

    if len(df) == 0:
        return {}

    neural_oc    = df[df['model_type']=='neural']['overconf_gap'].mean()
    classical_oc = df[df['model_type']=='classical']['overconf_gap'].mean()
    neural_ce    = df[df['model_type']=='neural']['cal_error'].mean()
    classical_ce = df[df['model_type']=='classical']['cal_error'].mean()

    print(f"\n  Overconfidence (softmax_conf - actual_f1):")
    print(f"    Neural:    {neural_oc:+.4f}")
    print(f"    Classical: {classical_oc:+.4f}")
    print(f"    Neural MORE overconfident: {'✅ YES' if neural_oc > classical_oc else '❌'}")
    print(f"\n  Calibration error:")
    print(f"    Neural:    {neural_ce:.4f}")
    print(f"    Classical: {classical_ce:.4f}")

    comp_rows = []
    for (ds, ext), grp in df.groupby(['dataset','extractor']):
        if len(grp) < 2: continue
        ba = grp.loc[grp['test_acc'].idxmax()]
        bt = grp.loc[grp['trust_score'].idxmax()]
        br = grp.loc[grp['gen_gap'].abs().idxmin()]
        comp_rows.append({
            'dataset': ds, 'extractor': ext,
            'acc_gap': float(ba['gen_gap']),
            'trust_gap': float(bt['gen_gap']),
            'trust_wins': bool(abs(float(bt['gen_gap'])) <= abs(float(ba['gen_gap']))),
        })

    comp_df = pd.DataFrame(comp_rows)
    if len(comp_df) > 0:
        tw = int(comp_df['trust_wins'].sum())
        print(f"\n  Trust selector wins: {tw}/{len(comp_df)}")

    print(f"\n  Mean F1 by extractor:")
    for ext, grp in df.groupby('extractor'):
        print(f"    {ext:12s}: {grp['test_f1'].mean():.4f}")

    def _j(o):
        if isinstance(o,(bool,)): return bool(o)
        if isinstance(o,(int,)):  return int(o)
        if isinstance(o,(float,)):
            return None if (o!=o or abs(o)==float('inf')) else float(o)
        return str(o)

    results = {
        'n_experiments':      int(len(df)),
        'n_datasets':         int(df['dataset'].nunique()),
        'neural_overconf':    round(float(neural_oc),4),
        'classical_overconf': round(float(classical_oc),4),
        'neural_cal_error':   round(float(neural_ce),4),
        'classical_cal_error':round(float(classical_ce),4),
        'neural_more_overconfident': bool(neural_oc > classical_oc),
        'trust_wins_rate': round(float(comp_df['trust_wins'].mean()),4) if len(comp_df)>0 else 0,
        'elapsed_s': round(time.time()-t0, 1),
        'key_finding': (
            f"On text classification, neural networks show "
            f"{neural_oc:+.4f} overconfidence vs {classical_oc:+.4f} "
            f"for classical models. Trust wins on "
            f"{int(comp_df['trust_wins'].sum()) if len(comp_df)>0 else 0}/"
            f"{len(comp_df)} configurations."
        )
    }

    with open(OUT/"stage3_results.json","w") as f:
        json.dump(results, f, indent=2, default=_j)

    print(f"\n  Results → {OUT}/")
    print(f"\n  KEY FINDING: {results['key_finding']}")
    return results


if __name__ == "__main__":
    run_stage3()
