import os 
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))


from utils.utils import load_config_file, save_model

import pandas as pd
import joblib
import structlog

import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from feature_engine.imputation import MeanMedianImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import RobustScaler, StandardScaler
from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser

from evaluation.classifiers_eval import ModelEvaluation


mlflow.set_tracking_uri(load_config_file().get('uri_tracking'))
mlflow.set_experiment(load_config_file().get('experiment'))

logger = structlog.getLogger()

class TrainModel():
    def __init__(self, dados_x: pd.DataFrame, dados_y: pd.DataFrame):
        self.dados_X = dados_x
        self.dados_y = dados_y
        self.model_name = load_config_file().get('model_name')
    
    def get_best_model(self):
        logger.info('Obtendo melhor modelo..')

        df_mlflow = mlflow.search_runs(filter_string='metrics.valid_roc_auc < 1').sort_values('metrics.valid_roc_auc', ascending=False)
        run_id = df_mlflow.loc[df_mlflow['metrics.valid_roc_auc'].idxmax()]['run_id']
        print(df_mlflow.columns)
        df_best_params = df_mlflow.loc[df_mlflow['run_id'] == run_id][['params.discretizer', 'params.warm_start', 'params.multi_class',
                                                                      'params.solver', 'params.tol', 'params.fit_intercept', 'params.C',
                                                                      'params.imputer', 'params.scaler', 'params.max_iter',
                                                                      'params.class_weight']]
        
        best_roc_auc = df_mlflow.loc[df_mlflow['metrics.valid_roc_auc'].idxmax()]['metrics.valid_roc_auc']
        return best_roc_auc, df_best_params
    
    def run(self):
        _, df_best_params = self.get_best_model()
        logger.info(f'Iniciando o treinamento do modelo: {self.model_name}')

        with mlflow.start_run(run_name='final_model'):
            mlflow.set_tag('model_name', self.model_name)
            model = LogisticRegression(warm_start=eval(df_best_params['params.warm_start'].values[0]),
                                       multi_class=df_best_params['params.multi_class'].values[0],
                                       solver=df_best_params['params.solver'].values[0],
                                       tol=eval(df_best_params['params.tol'].values[0]),
                                        fit_intercept=eval(df_best_params['params.fit_intercept'].values[0]),
                                        C=eval(df_best_params['params.C'].values[0]),
                                        max_iter=eval(df_best_params['params.max_iter'].values[0]),
                                        class_weight=eval(df_best_params['params.class_weight'].values[0]) )
            
            pipe = Pipeline([
                ('imputer', eval(df_best_params['params.imputer'].values[0])),
                ('discretizer', eval(df_best_params['params.discretizer'].values[0])),
                ('scaler', eval(df_best_params['params.discretizer'].values[0])),
                ('model', model)
            ])

            pipe.fit(self.dados_X, self.dados_y)
            # logar metricas de avaliacao
            y_val_preds = pipe.predict_proba(self.dados_X)[:, 1]
            model_eval = ModelEvaluation(model, self.dados_X, self.dados_y)

            val_roc_auc = model_eval.evaluate_predictions(self.dados_y, y_val_preds)
            mlflow.log_metric('valid_roc_auc', val_roc_auc)

            #registrar modelo
            mlflow.sklearn.log_model(pipe, 
                                     self.model_name, 
                                     pyfunc_predict_fn='predict_proba', 
                                     input_example=self.dados_X.iloc[[0]], 
                                     registered_model_name=self.model_name)
            
            
