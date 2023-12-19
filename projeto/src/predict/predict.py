import json
import sqlite3

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import requests
import structlog

conn = sqlite3.connect("../../preds.db")
cursor = conn.cursor()
logger = structlog.getLogger()


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("prob_loan")


class Predict:
    def __init__(self, dataframe: pd.DataFrame):
        self.datarame = dataframe
        self.endpoint = "http://127.0.0.1:5001/invocations"

    def run(self):
        logger.info("Iniciando a predicao...")
        to_inference = {
            "dataframe_split": {
                "columns": self.datarame.columns.to_list(),
                "data": self.datarame.replace(np.nan, None).values.tolist(),
            }
        }

        response = requests.post(self.endpoint, json=to_inference)

        logger.info("Predicoes Finalizadas")

        probabilidades = np.array(json.loads(response.text).get("predictions", []))[:, 1]
        df_probs = self._results(probabilidades)

        self._capture_inputs_and_predictions(to_inference, df_probs)
        logger.info("Resultados salvos na base de dados")

        return df_probs

    def _capture_inputs_and_predictions(self, input, preds):
        input_df = pd.DataFrame(
            input["dataframe_split"]["data"],
            columns=input["dataframe_split"]["columns"],
        )
        input_df["Preds_Prob"] = preds
        self._store_in_database(input_df)

    # Armazenar no banco de dados
    def _store_in_database(self, input_df):
        input_df.to_sql("predictions", conn, if_exists="append", index=False)
        conn.commit()
        conn.close()

    def _results(self, probabilidades: np.array):
        df_results = pd.DataFrame()
        logger.info("Salvando as Probabilidades")
        df_results["probabilidade"] = probabilidades
        return df_results
