
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MarkowitzPortfolio:

    @staticmethod
    def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
        portfolio_return = float(weights @ mean_returns)
        portfolio_variance = float(weights @ cov_matrix @ weights)

        portfolio_volatility = np.sqrt(max(portfolio_variance, 0.0))
        return portfolio_return, portfolio_volatility

    @staticmethod
    def calculate_sharpe_ratio(portfolio_return, risk_free_rate, portfolio_volatility):
        if portfolio_volatility <= 1e-12:
            return np.nan

        return (portfolio_return - risk_free_rate) / portfolio_volatility

    def efficient_frontier(
        self, mean_returns, cov_matrix,
        risk_free_rate=0.04,
        num_portfolios=10000,
        frontier_points=60,
        seed=42
    ):
        """
        Optimise long-only portfolios without drawing anything, so it can run off the GUI thread.

        Inputs must be annualized: a pandas Series of mean returns and the covariance DataFrame
        of the same assets.

        Returns a dict with the maximum Sharpe and minimum variance portfolios (weights, return,
        volatility), the efficient frontier (volatility, return) and randomly simulated
        portfolios (volatility, return, sharpe).
        """
        assets = mean_returns.index
        mu = mean_returns.to_numpy(dtype=float)
        cov = cov_matrix.loc[assets, assets].to_numpy(dtype=float)

        n = len(mu)

        if n < 2 or not np.isfinite(mu).all() or not np.isfinite(cov).all():
            raise ValueError("Provide at least 2 assets and finite inputs.")

        if not np.allclose(cov, cov.T):
            raise ValueError("Covariance matrix must be symmetric.")

        if np.linalg.eigvalsh(cov).min() < -1e-10:
            raise ValueError("Covariance matrix must be positive semidefinite.")

        initial_weights = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n

        budget_constraint = {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0
        }

        def performance(w):
            return self.calculate_portfolio_performance(w, mu, cov)

        def negative_sharpe(w):
            ret, vol = performance(w)
            if vol <= 1e-12:
                return 1e10
            return -(ret - risk_free_rate) / vol

        # 1. Optimize maximum Sharpe portfolio
        max_sharpe = minimize(
            negative_sharpe,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=[budget_constraint],
            options={"maxiter": 1000, "ftol": 1e-10}
        )

        # 2. Optimize global minimum variance portfolio
        min_variance = minimize(
            lambda w: w @ cov @ w,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=[budget_constraint],
            options={"maxiter": 1000, "ftol": 1e-10}
        )

        if not max_sharpe.success or not min_variance.success:
            raise RuntimeError("Portfolio optimization did not converge.")

        min_return, min_vol = performance(min_variance.x)
        max_return, max_vol = performance(max_sharpe.x)

        # 3. Calculate the true efficient frontier
        target_returns = np.linspace(
            min_return, mu.max(), frontier_points
        )

        frontier = []

        for target in target_returns:
            constraints = [
                budget_constraint,
                {
                    "type": "eq",
                    "fun": lambda w, t=target: w @ mu - t
                }
            ]

            result = minimize(
                lambda w: w @ cov @ w,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-10}
            )

            if result.success:
                _, vol = performance(result.x)
                frontier.append((vol, target))

        frontier = pd.DataFrame(
            frontier, columns=["volatility", "return"]
        )

        # 4. Simulate feasible portfolios
        rng = np.random.default_rng(seed)
        simulated_weights = rng.dirichlet(
            np.ones(n), size=num_portfolios
        )

        simulated_returns = simulated_weights @ mu
        variances = np.einsum(
            "ij,jk,ik->i",
            simulated_weights, cov, simulated_weights
        )
        simulated_vols = np.sqrt(np.maximum(variances, 0))

        simulated_sharpes = np.divide(
            simulated_returns - risk_free_rate,
            simulated_vols,
            out=np.full(num_portfolios, np.nan),
            where=simulated_vols > 1e-12
        )

        # 5. Return reusable results
        return {
            "max_sharpe_weights": pd.Series(
                max_sharpe.x, index=assets, name="weight"
            ),
            "max_sharpe_return": max_return,
            "max_sharpe_volatility": max_vol,
            "max_sharpe_ratio": -max_sharpe.fun,
            "min_variance_weights": pd.Series(
                min_variance.x, index=assets, name="weight"
            ),
            "min_variance_return": min_return,
            "min_variance_volatility": min_vol,
            "efficient_frontier": frontier,
            "simulated_portfolios": pd.DataFrame({
                "volatility": simulated_vols,
                "return": simulated_returns,
                "sharpe": simulated_sharpes,
            }),
        }

    def plot_efficient_frontier(
        self, mean_returns, cov_matrix,
        risk_free_rate=0.04,
        num_portfolios=10000,
        frontier_points=60,
        seed=42
    ):
        # Imported here so the desktop app, which draws its own chart, never touches pyplot.
        import matplotlib.pyplot as plt

        results = self.efficient_frontier(
            mean_returns, cov_matrix, risk_free_rate,
            num_portfolios, frontier_points, seed
        )
        frontier = results["efficient_frontier"]
        simulated = results["simulated_portfolios"]

        fig, ax = plt.subplots(figsize=(11, 7))

        points = ax.scatter(
            simulated["volatility"],
            simulated["return"],
            c=simulated["sharpe"],
            cmap="viridis",
            s=8,
            alpha=0.35
        )

        ax.plot(
            frontier["volatility"],
            frontier["return"],
            color="orange",
            linewidth=2.5,
            label="Efficient Frontier"
        )

        ax.scatter(
            results["max_sharpe_volatility"], results["max_sharpe_return"],
            marker="*", s=300, color="red",
            label="Maximum Sharpe"
        )

        ax.scatter(
            results["min_variance_volatility"], results["min_variance_return"],
            marker="D", s=100, color="green",
            label="Minimum Variance"
        )

        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Annualized Expected Return")
        ax.set_title("Markowitz Portfolio Optimization")
        ax.legend()
        fig.colorbar(points, ax=ax, label="Sharpe Ratio")

        return {**results, "figure": fig}
