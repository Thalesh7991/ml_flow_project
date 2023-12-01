from pandera import Check, Column, DataFrameSchema
import pandas as pd

class DataValidation:
    def __init__(self, columns_to_use) -> None:
        self.columns_to_use = columns_to_use

    def check_shape_data(self, dataframe: pd.DataFrame)-> bool:
        try:
            print('Validação iniciou..')
            dataframe.columns = self.columns_to_use
            return True
        except Exception as e:
            print(f'Validação errou: {e}')
            return False
    
    def check_columns(self,dataframe: pd.DataFrame)-> bool:
        schema = DataFrameSchema(
                {
                    "target": Column(int, Check.isin([0, 1]), Check(lambda x: x > 0), coerce=True),
                    "TaxaDeUtilizacaoDeLinhasNaoGarantidas": Column(float, nullable=True),
                    "Idade": Column(int, nullable=True),
                    "NumeroDeVezes30-59DiasAtrasoNaoPior": Column(int, nullable=True),
                    "TaxaDeEndividamento": Column(float, nullable=True),
                    "RendaMensal": Column(float, nullable=True),
                    "NumeroDeLinhasDeCreditoEEmprestimosAbertos": Column(int, nullable=True),
                    "NumeroDeVezes90DiasAtraso": Column(int, nullable=True),
                    "NumeroDeEmprestimosOuLinhasImobiliarias": Column(int, nullable=True),
                    "NumeroDeVezes60-89DiasAtrasoNaoPior": Column(int, nullable=True),
                    "NumeroDeDependentes": Column(float, nullable=True)
                }
            )
        try:
            schema.validate(dataframe)
            print("Validation columns passed...")
            return True
        except pandera.errors.SchemaErrors as exc:
            print("Validation columns failed...")
            pandera.display(exc.failure_cases)
        return False
    
    def run(self, dataframe: pd.DataFrame) -> bool:
        if self.check_shape_data(dataframe) and self.check_columns(dataframe):
            print('Validacao com sucesso.')
            return True 
        else:
            print('Validacao falhou.')
            return False

