class Critic:
    def score(self, res: dict) -> float:
        auc = float(res.get("auc", 0.0))
        acc = float(res.get("avg_acc", 0.0))
        rmse = float(res.get("avg_rmse", 0.0) or 0.0)
        return 2.0*auc + 1.0*acc - 0.02*rmse
