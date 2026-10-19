
import numpy as np
import pandas as pd

from arch import arch_model
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SVRGARCH:
    """
    Hybrid SVR-GARCH(1,1) volatility forecasting model.

    GARCH fit:  first 60% of observations
    SVR fit:    next 20% of observations
    Test:       final 20% of observations

    Returns should preferably be daily percentage returns.
    """

    def __init__(
        self,
        data,
        features=None,
        garch_fraction=0.60,
        svr_end_fraction=0.80,
        C=5.0,
        epsilon=0.1,
        kernel="rbf",
        delta=1e-6
    ):
        self.returns = pd.Series(data, dtype=float).copy()
        self.features = features

        self.n = len(self.returns)
        self.n0 = int(garch_fraction * self.n)
        self.n1 = int(svr_end_fraction * self.n)

        if not 0 < self.n0 < self.n1 < self.n:
            raise ValueError("Invalid chronological split.")

        if self.n0 < 30 or self.n1 - self.n0 < 10:
            raise ValueError("Not enough observations for fitting.")

        if not np.isfinite(self.returns.to_numpy()).all():
            raise ValueError("Returns contain NaN or infinite values.")

        self.svr_params = {
            "kernel": kernel,
            "C": C,
            "epsilon": epsilon
        }
        self.delta = delta
        self.is_fitted = False

    def _make_features(self):
        """
        Build features using only information known
        before the return being forecast.
        """
        if self.features is not None:
            if not self.features.index.equals(self.returns.index):
                raise ValueError("Feature and return indices must match.")
            X = self.features.copy()
        else:
            r = self.returns

            X = pd.DataFrame(index=r.index)
            X["return_lag1"] = r.shift(1)
            X["return_lag2"] = r.shift(2)
            X["abs_return_lag1"] = r.abs().shift(1)
            X["rv_5"] = r.pow(2).rolling(5).mean().shift(1)
            X["rv_20"] = r.pow(2).rolling(20).mean().shift(1)

        return X

    def fit(self):
        # -----------------------------------
        # Step 1: Fit GARCH on first 60% only
        # -----------------------------------
        train_returns = self.returns.iloc[:self.n0]

        garch = arch_model(
            train_returns,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="t",
            rescale=False
        )

        self.garch_fit_ = garch.fit(disp="off")

        params = self.garch_fit_.params
        omega = float(params["omega"])
        alpha = float(params["alpha[1]"])
        beta = float(params["beta[1]"])

        # -----------------------------------
        # Step 2: Recursive one-step forecasts
        # -----------------------------------
        h = np.full(self.n, np.nan)

        previous_var = float(
            self.garch_fit_.conditional_volatility.iloc[-1] ** 2
        )
        previous_return = float(self.returns.iloc[self.n0 - 1])

        for i in range(self.n0, self.n):
            next_var = (
                omega
                + alpha * previous_return**2
                + beta * previous_var
            )

            h[i] = max(next_var, self.delta)

            # Update only after making the forecast
            previous_return = self.returns.iloc[i]
            previous_var = h[i]

        self.garch_variance_ = h

        # -----------------------------------
        # Step 3: Prepare SVR features
        # -----------------------------------
        X = self._make_features()

        X["log_garch_variance"] = np.log(
            np.maximum(h, self.delta)
        )

        X = X.astype(float)

        if not np.isfinite(X.iloc[self.n0:].to_numpy()).all():
            raise ValueError(
                "SVR features contain missing or infinite values."
            )

        self.X_ = X

        # -----------------------------------
        # Step 4: SVR correction target
        # -----------------------------------
        r_svr = self.returns.iloc[self.n0:self.n1].to_numpy()
        h_svr = h[self.n0:self.n1]

        correction_target = np.log(
            (r_svr**2 + self.delta)
            / (h_svr + self.delta)
        )

        X_train = X.iloc[self.n0:self.n1]

        # -----------------------------------
        # Step 5: Train SVR
        # -----------------------------------
        self.svr_ = make_pipeline(
            StandardScaler(),
            SVR(**self.svr_params)
        )

        self.svr_.fit(X_train, correction_target)

        # -----------------------------------
        # Step 6: Calibrate hybrid variance
        # -----------------------------------
        correction = self.svr_.predict(X_train)

        train_uncalibrated = (
            h_svr * np.exp(np.clip(correction, -30, 30))
        )

        self.calibration_ = (
            max(np.mean(r_svr**2), self.delta)
            / max(np.mean(train_uncalibrated), self.delta)
        )

        self.is_fitted = True
        return self

    def predict(self, split="test"):
        """
        Return GARCH and hybrid forecasts.

        split='test': final 20%
        split='svr':  SVR training segment
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        if split == "test":
            start, end = self.n1, self.n
        elif split == "svr":
            start, end = self.n0, self.n1
        else:
            raise ValueError("split must be 'test' or 'svr'.")

        X = self.X_.iloc[start:end]
        garch_var = self.garch_variance_[start:end]

        # SVR correction
        correction = self.svr_.predict(X)

        # Hybrid conditional variance
        hybrid_var = (
            self.calibration_
            * garch_var
            * np.exp(np.clip(correction, -30, 30))
        )

        hybrid_var = np.maximum(hybrid_var, self.delta)

        return pd.DataFrame(
            {
                "GARCH_variance": garch_var,
                "GARCH_SVR_variance": hybrid_var,
                "GARCH": np.sqrt(garch_var),
                "GARCH_SVR": np.sqrt(hybrid_var)
            },
            index=self.returns.index[start:end]
        )

    @staticmethod
    def qlike(proxy, variance):
        v = np.maximum(np.asarray(variance), 1e-10)
        proxy = np.asarray(proxy)

        return np.mean(np.log(v) + proxy / v)

    def evaluate(self):
        """
        Evaluate out-of-sample variance forecasts.
        Lower QLIKE and MSE are better.
        """
        predictions = self.predict(split="test")

        realized = (
            self.returns.iloc[self.n1:].to_numpy() ** 2
        )

        scores = {}

        for model in ["GARCH", "GARCH_SVR"]:
            variance = predictions[f"{model}_variance"].to_numpy()

            scores[model] = {
                "QLIKE": self.qlike(realized, variance),
                "MSE": np.mean((realized - variance) ** 2)
            }

        return pd.DataFrame(scores).T
