import json
import boto3 
import pandas as pd


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../src"))
from utils.utils import load_config_file
from data.data_load import DataLoad

app_name = load_config_file().get("app_name")
region = os.environ.get('REGION')


dt = DataLoad()
# #USAR FUNÇÃO PARA CARREGAMENTO DE DADOS
df_test = pd.read_csv('C:/Users/thale/Documents/Projetos_DS/ml_flow/ml_flow_project/projeto/data/raw/test.csv')

def query(input_json):
    client = boto3.session.Session().client('sagemaker-runtime', region)
    response = client.invoke_endpoint(EndpointName=app_name,Body=input_json,ContentType='application/json')

    preds = response['Body'].read().decode('ascii')
    preds = json.loads(preds)
    print(f'Resposta Recebida: {preds}')
    return preds

# # Manupulaçaõ dos dados 
data = {"dataframe_split": df_test.iloc[[0]].to_dict(orient='split')}
byte_data = json.dumps(data).encode('utf-8')

output = query(byte_data)

resp = pd.DataFrame([output])
print(resp)


