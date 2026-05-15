
import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.validation import check_X_y, check_array
from scipy.special import softmax, expit
warnings.filterwarnings("ignore")

class DeepMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=(256,128,64,32), max_iter=300, random_state=42):
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_
        self._mlp = MLPClassifier(hidden_layer_sizes=self.hidden_layers,
            max_iter=self.max_iter, random_state=self.random_state,
            early_stopping=False)
        self._mlp.fit(X, y_enc)
        return self

    def predict_proba(self, X):
        return self._mlp.predict_proba(check_array(X))

    def predict(self, X):
        return self._le.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class CNN1DTabularClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_filters=32, kernel_size=3, dense_units=64,
                 lr=0.01, epochs=60, batch_size=32, random_state=42):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_
        n_classes = len(self.classes_)
        n, p = X.shape
        rng = np.random.RandomState(self.random_state)
        k = min(self.kernel_size, p)
        f = self.n_filters
        d = self.dense_units
        # Weights
        self._W1 = rng.randn(f, k) * np.sqrt(2.0/k)
        self._b1 = np.zeros(f)
        self._Wd = rng.randn(f, d) * np.sqrt(2.0/f)
        self._bd = np.zeros(d)
        self._Wo = rng.randn(d, n_classes) * np.sqrt(2.0/d)
        self._bo = np.zeros(n_classes)

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                b_idx = idx[s:s+self.batch_size]
                Xb, yb = X[b_idx], y_enc[b_idx]
                dWo = np.zeros_like(self._Wo)
                dbo = np.zeros_like(self._bo)
                dWd = np.zeros_like(self._Wd)
                dbd = np.zeros_like(self._bd)
                for xi, yi in zip(Xb, yb):
                    # conv: slide kernel over features
                    L = p - k + 1
                    conv = np.array([np.dot(self._W1[fi], xi[i:i+k]) + self._b1[fi]
                                     for fi in range(f) for i in range(L)]).reshape(f, L)
                    conv = np.maximum(conv, 0)
                    gmp = conv.max(axis=1)  # (f,)
                    h = np.maximum(gmp @ self._Wd + self._bd, 0)
                    logits = h @ self._Wo + self._bo
                    p_out = softmax(logits)
                    delta = p_out.copy(); delta[yi] -= 1
                    dWo += np.outer(h, delta)
                    dbo += delta
                    dh = (self._Wo @ delta) * (h > 0)
                    dWd += np.outer(gmp, dh)
                    dbd += dh
                bs = max(len(b_idx), 1)
                self._Wo -= self.lr * dWo / bs
                self._bo -= self.lr * dbo / bs
                self._Wd -= self.lr * dWd / bs
                self._bd -= self.lr * dbd / bs
        return self

    def predict_proba(self, X):
        X = self._scaler.transform(check_array(X))
        n, p = X.shape
        k = min(self.kernel_size, p)
        f = self.n_filters
        out = []
        for xi in X:
            L = p - k + 1
            if L <= 0: L = 1
            conv = np.zeros((f, max(L,1)))
            for fi in range(f):
                for i in range(max(L,1)):
                    end = min(i+k, p)
                    seg = xi[i:end]
                    w_seg = self._W1[fi, :len(seg)]
                    conv[fi, i] = np.dot(w_seg, seg) + self._b1[fi]
            conv = np.maximum(conv, 0)
            gmp = conv.max(axis=1)
            h = np.maximum(gmp @ self._Wd + self._bd, 0)
            logits = h @ self._Wo + self._bo
            out.append(softmax(logits))
        return np.array(out)

    def predict(self, X):
        return self._le.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class LSTMTabularClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden=32, n_timesteps=4, lr=0.01,
                 epochs=50, batch_size=32, random_state=42):
        self.n_hidden = n_hidden
        self.n_timesteps = n_timesteps
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _reshape(self, X):
        n, p = X.shape
        t = self.n_timesteps
        fp = max(1, p // t)
        total = t * fp
        if total > p:
            X = np.pad(X, ((0,0),(0,total-p)))
        else:
            X = X[:, :total]
        self._fp = fp
        return X.reshape(n, t, fp)

    def _lstm_step(self, x_t, h, c, Wf,Uf,bf, Wi,Ui,bi, Wc,Uc,bc, Wo,Uo,bo):
        f = expit(x_t @ Wf + h @ Uf + bf)
        i = expit(x_t @ Wi + h @ Ui + bi)
        cc = np.tanh(x_t @ Wc + h @ Uc + bc)
        o = expit(x_t @ Wo + h @ Uo + bo)
        c_new = f*c + i*cc
        h_new = o*np.tanh(c_new)
        return h_new, c_new

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_
        nc = len(self.classes_)
        X_seq = self._reshape(X)
        fp = self._fp
        h = self.n_hidden
        rng = np.random.RandomState(self.random_state)
        s = lambda a,b: rng.randn(a,b)*np.sqrt(2.0/(a+b))
        self._Wf=s(fp,h); self._Uf=s(h,h); self._bf=np.zeros(h)
        self._Wi=s(fp,h); self._Ui=s(h,h); self._bi=np.zeros(h)
        self._Wc=s(fp,h); self._Uc=s(h,h); self._bc=np.zeros(h)
        self._Wo=s(fp,h); self._Uo=s(h,h); self._bo=np.zeros(h)
        self._Wout=s(h,nc); self._bout=np.zeros(nc)
        n = len(X_seq)
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for st in range(0, n, self.batch_size):
                bidx = idx[st:st+self.batch_size]
                dWout=np.zeros_like(self._Wout); dbout=np.zeros_like(self._bout)
                for xi, yi in zip(X_seq[bidx], y_enc[bidx]):
                    hh=np.zeros(h); cc=np.zeros(h)
                    for t in range(self.n_timesteps):
                        hh,cc=self._lstm_step(xi[t],hh,cc,
                            self._Wf,self._Uf,self._bf,
                            self._Wi,self._Ui,self._bi,
                            self._Wc,self._Uc,self._bc,
                            self._Wo,self._Uo,self._bo)
                    logits=hh@self._Wout+self._bout
                    p_out=softmax(logits); delta=p_out.copy(); delta[yi]-=1
                    dWout+=np.outer(hh,delta); dbout+=delta
                bs2=max(len(bidx),1)
                self._Wout-=self.lr*dWout/bs2
                self._bout-=self.lr*dbout/bs2
        return self

    def predict_proba(self, X):
        X = self._scaler.transform(check_array(X))
        n, p = X.shape
        fp = self._fp
        t = self.n_timesteps
        total = t * fp
        if total > p:
            X = np.pad(X, ((0,0),(0,total-p)))
        else:
            X = X[:, :total]
        X_seq = X.reshape(n, t, fp)
        h_dim = self.n_hidden
        out = []
        for xi in X_seq:
            hh=np.zeros(h_dim); cc=np.zeros(h_dim)
            for step in range(t):
                hh,cc=self._lstm_step(xi[step],hh,cc,
                    self._Wf,self._Uf,self._bf,
                    self._Wi,self._Ui,self._bi,
                    self._Wc,self._Uc,self._bc,
                    self._Wo,self._Uo,self._bo)
            logits=hh@self._Wout+self._bout
            out.append(softmax(logits))
        return np.array(out)

    def predict(self, X):
        return self._le.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


NEURAL_MODELS = {
    "deep_mlp":     DeepMLPClassifier(hidden_layers=(256,128,64,32), max_iter=300),
    "cnn1d_tabular": CNN1DTabularClassifier(n_filters=16, kernel_size=3, epochs=40),
    "lstm_tabular":  LSTMTabularClassifier(n_hidden=32, n_timesteps=4, epochs=40),
}

def get_all_neural_models():
    from sklearn.base import clone
    return {k: clone(v) for k, v in NEURAL_MODELS.items()}

def get_neural_models(clone_all=True):
    from sklearn.base import clone
    if clone_all:
        return {k: clone(v) for k, v in NEURAL_MODELS.items()}
    return NEURAL_MODELS.copy()

def get_combined_registry(include_classical=True):
    from sklearn.base import clone
    from src.models.model_registry import get_all_models
    combined = get_neural_models()
    if include_classical:
        combined.update(get_all_models(enabled_only=True))
    return combined
