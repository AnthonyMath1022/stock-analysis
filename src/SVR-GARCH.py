from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from arch import arch_model

class svr_garch:

    def __init__(self, data):
        self.data = data
        n = len(data)
        self.n0 = int(0.60 * n)
        self.n1 = int(0.80 * n)


    def fit(self, data):
        garch = arch_model(
        self.data,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False)

        self.fit = garch.fit(disp="off")

        self.omega = self.fit.params["omega"]
        self.alpha = self.fit.params["alpha[1]"]
        self.beta = self.fit.params["beta[1]"]

    def predict():




