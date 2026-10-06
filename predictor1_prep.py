
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder

SEED = 42


class RadiopharmPreprocessor(BaseEstimator, TransformerMixin):


    def __init__(self, cat_cols=None, num_cols=None, derived=False, smooth='auto',
                 random_state=SEED):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.derived = derived
        self.smooth = smooth
        self.random_state = random_state

    def _coerce(self, X):
        if isinstance(X, pd.DataFrame):
            return X[list(self.cat_cols) + list(self.num_cols)]
        return pd.DataFrame(np.asarray(X), columns=self.all_cols_)

    def _cat_frame(self, X):
        return X[list(self.cat_cols)].fillna('NA').astype(str)

    def _num_frame(self, X):
        Xn = X[list(self.num_cols)].copy()
        for c in Xn.columns:
            Xn[c] = pd.to_numeric(Xn[c], errors='coerce')
        return Xn

    def _build_numeric(self, Xn):

        arr = self.scaler_.transform(self.imputer_.transform(Xn.values))

        out = pd.DataFrame(arr, columns=list(self.num_cols), index=Xn.index)
        if not self.derived:
            return out
        nc = list(self.num_cols)
        for i in range(len(nc)):
            for j in range(i + 1, min(i + 3, len(nc))):
                out[f'{nc[i]}_mul_{nc[j]}'] = out[nc[i]] * out[nc[j]]
        for c in nc[:3]:
            out[f'{c}_sq'] = out[c] ** 2
            out[f'{c}_log'] = np.log1p(np.abs(out[c]))
        if len(nc) >= 2:
            out[f'{nc[0]}_div_{nc[1]}'] = out[nc[0]] / (out[nc[1]] + 1e-6)
        return out

    def fit(self, X, y=None):
        X = self._coerce(X)
        self.all_cols_ = list(self.cat_cols) + list(self.num_cols)
        yv = np.asarray(y, dtype=float).ravel()
        self.te_ = TargetEncoder(random_state=self.random_state, smooth=self.smooth)
        self.te_.fit(self._cat_frame(X), yv)
        Xn = self._num_frame(X)
        self.imputer_ = SimpleImputer(strategy='median', missing_values=np.nan)
        self.imputer_.fit(Xn.values)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(self.imputer_.transform(Xn.values))
        built = self._build_numeric(Xn)
        self.feature_names_out_ = list(self.cat_cols) + list(built.columns)
        return self

    def transform(self, X):
        X = self._coerce(X)
        enc = np.asarray(self.te_.transform(self._cat_frame(X)), dtype=float)
        Xc = pd.DataFrame(enc, columns=list(self.cat_cols), index=X.index)
        Xn = self._build_numeric(self._num_frame(X))
        return pd.concat([Xc, Xn], axis=1).fillna(0)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_)


def log_forward(y, kind='ln1p'):

    y = np.asarray(y, dtype=float)
    if kind == 'ln1p':
        return np.log1p(y)
    if kind == 'log10':
        return np.log10(y)
    if kind == 'log10p':
        return np.log10(1.0 + y)
    raise ValueError(kind)


def log_inverse(z, kind='ln1p'):

    z = np.clip(np.asarray(z, dtype=float), -50.0, 50.0)
    if kind == 'ln1p':
        return np.clip(np.expm1(z), 0.0, None)
    if kind == 'log10':
        return np.power(10.0, z)
    if kind == 'log10p':
        return np.maximum(np.power(10.0, z) - 1.0, 0.0)
    raise ValueError(kind)
