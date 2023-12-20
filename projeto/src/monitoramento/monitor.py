import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))




import sqlite3
from evidently.report import Report
from evidently.metrics import *
from evidently.metric_preset import DataDriftPreset
from evidently.test_preset import DataDriftTestPreset
import pandas as pd 

from data.data_load import DataLoad



class ModelMonitoring():
    def __init__(self):
        self.query_pred = "SELECT * FROM predictions"
    
    def get_pred_data(self):
        conn = sqlite3.connect("C:/Users/thale/Documents/Projetos_DS/ml_flow/ml_flow_project/preds.db")
        df_pred = pd.read_sql_query(self.query_pred, conn)
        conn.close()
        return df_pred 
    
    def get_training_data(self):
        dl = DataLoad()
        df = dl.load_data('train_dataset_name')
        return df
    
    def run(self):
        df_cur = self.get_pred_data()
        def_ref = self.get_training_data().drop('target', axis=1)

        docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../docs'))
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)

        model_card = Report(metrics=[
            DatasetSummaryMetric(),  
            DataDriftPreset(),
            DatasetMissingValuesMetric()
        ])

        model_card.run(reference_data=def_ref, current_data=df_cur)
        
        model_card.save_html(os.path.join(docs_dir, "model_monitoring_report.html"))

mm = ModelMonitoring()
mm.run()