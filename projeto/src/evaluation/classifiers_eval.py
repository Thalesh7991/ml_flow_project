import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import structlog
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = structlog.getLogger()

from utils.utils import load_config_file


class ModelEvaluation:
    def __init__(self, model, x, y, n_splits=5):
        self.model = model
        self.dados_x = x
        self.dados_y = y
        self.n_splits = n_splits

    def cross_val_evaluate(self):
        logger.info("Iniciando a Cross Validation...")
        skf = StratifiedKFold(
            self.n_splits,
            shuffle=True,
            random_state=load_config_file().get("random_state"),
        )

        scores = cross_val_score(
            self.model, self.dados_x, self.dados_y, cv=skf, scoring="roc_auc"
        )

        return scores

    def roc_auc_scorer(self, model, x, y):
        y_pred = model.predict_proba(x)[:, 1]
        return roc_auc_score(y, y_pred)

    @staticmethod
    def evaluate_predictions(y_true, y_pred_proba):
        return roc_auc_score(y_true, y_pred_proba)

    def eval_metrics(self, dados_reais, dados_preditos):
        try:
            roc_auc = roc_auc_score(dados_reais, dados_preditos)
            precision = precision_score(dados_reais, dados_preditos)
            recall = recall_score(dados_reais, dados_preditos)
            f1 = f1_score(dados_reais, dados_preditos)
            classification_rep = classification_report(dados_reais, dados_preditos)
            confusion_mat = confusion_matrix(dados_reais, dados_preditos)

            return {
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "classification_report": classification_rep,
                "confusion_matrix": confusion_mat.tolist(),  # Converte a matriz para lista para ser serializável
            }
        except Exception as e:
            return f"Erro ao calcular métricas: {e}"
