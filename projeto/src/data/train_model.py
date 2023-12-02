import os 
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from utils.utils import load_config_file

import pandas as pd
import joblib
class TrainModel():
    def __init__(self, dados_x: pd.DataFrame, dados_y: pd.DataFrame):
        self.dados_X = dados_x
        self.dados_y = dados_y
        self.model_name = load_config_file().get('model_name')
    
    def train(self, model):
        model.fit(self.dados_X, self.dados_y)
        joblib.dump(model, self.model_name)
        return model
    

    
    # def predict(self, dados_predict: pd.DataFrame):
    #     model_trained = self._load_model()
    #     y_pred = model_trained.predict_proba(dados_predict)
    #     return y_pred
    
    # def _load_model(self):
    #     model = joblib.load(self.model_name)
    #     return model

        
