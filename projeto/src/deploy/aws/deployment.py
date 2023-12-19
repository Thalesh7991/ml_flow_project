

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../src"))


from mlflow.deployments import get_deploy_client
from utils.utils import load_config_file

from dotenv import load_dotenv

app_name = load_config_file().get("app_name")
arn = os.environ.get('ARN_ROLE') #arn do role
image_ecr_uri = os.environ.get('IMAGE_ECR_URI') # ECR
region = os.environ.get('REGION')

model_uri = "C:/Users/thale/Documents/Projetos_DS/ml_flow/ml_flow_project/mlartifacts/1/54e7a10302204051acd6dc0ef53a70dd/artifacts/modelo.joblib"

config = dict(
    execution_role_arn=arn,
    bucket_name="New-s3-bucket",
    image_url=image_ecr_uri,
    region_name=region,
    archive=False,
    instance_type="ml.m4.xlarge",
    instance_count=1,
    synchronous=True,
    timeout_seconds=3600,
    variant_name="prod-variant-2",
    tags={"training_timestamp": "2023-12-19"},
)

client = get_deploy_client("sagemaker")
deploy_client = client.create_deployment(app_name, model_uri=model_uri, flavor='python_function',config=config)

print(f'deploy_client: {deploy_client}')









