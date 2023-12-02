import pandas as pd
from sklearn.pipeline import Pipeline

import structlog
logger = structlog.getLogger()


class DataPreprocess():
    def __init__(self, pipe: Pipeline):
        self.pipe = pipe
        self.trained_pipe = None
    
    def train(self, dataframe: pd.DataFrame):
        logger.info('Iniciando o processamento')
        self.trained_pipe = self.pipe.fit(dataframe)
        return self.trained_pipe
    
    def transform(self, dataframe: pd.DataFrame):
        if self.trained_pipe is None:
            raise ValueError('O Pipeline não foi treinado...')
        logger.info('Iniciando a Transformação')
        transformed_data = self.trained_pipe.transform(dataframe)
        return transformed_data
    

    def pipeline(self):
        train_pipe = self.pipe
        train_pipe.fit(self.dataframe)
        return train_pipe
    
    def run(self):
        trained_pipeline = self.pipeline()
        data_processed = trained_pipeline.transform(self.dataframe)
        return data_processed
