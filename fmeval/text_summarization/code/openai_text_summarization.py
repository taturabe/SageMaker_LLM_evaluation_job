import os
import json
import argparse
from fmeval.data_loaders.data_config import DataConfig
from fmeval.model_runners.model_runner import ModelRunner

from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval.eval_algorithms.summarization_accuracy import SummarizationAccuracy
import boto3
import base64
from botocore.exceptions import ClientError
import requests
from dataclasses import dataclass
from typing import Tuple, Optional

from fmeval.model_runners.model_runner import ModelRunner

SECRET_NAME="OPENAI-API-KEY"
KEY_NAME = "openai_api_key"

parser = argparse.ArgumentParser()

parser.add_argument('--input_data', type=str, required=True, help='filename placed in ProcessingInput')
parser.add_argument('--model_id', type=str, required=True, help='model id')


args = parser.parse_args()


def get_secret():
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=SECRET_NAME
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e

    # Decrypts secret using the associated KMS CMK.
    # Depending on whether the secret is a string or binary, one of these fields will be populated.
    return get_secret_value_response['SecretString']


@dataclass
class ChatGPTModelConfig:
    temperature: float
    top_p: float
    max_tokens: int
    api_key: str

class ChatGPTModelRunner(ModelRunner):
    url = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, model_config: ChatGPTModelConfig):
        self.config = model_config
        
    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        payload = json.dumps({
            "model": args.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "n": 1,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 0
        })
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + self.config.api_key
        }
        
        response = requests.request("POST", self.url, headers=headers, data=payload)
        print("#################")
        print(response.text)
        return json.loads(response.text)["choices"][0]["message"]["content"], None





input_data_path = os.path.join("/opt/ml/processing/input/data", args.input_data)




try:
    os.makedirs("/opt/ml/processing/output/evaluation")
    print("Successfully created directories")
except Exception as e:
    # if the Processing call already creates these directories (or directory otherwise cannot be created)
    print(e)
    print("Could not make directories")
    pass

data_config = DataConfig(
    dataset_name="xsum_sample",
    dataset_uri=input_data_path,
    dataset_mime_type=MIME_TYPE_JSONLINES,
    model_input_location="document",
    target_output_location="summary"
)


# Note: Don't forget to include your api_key
model_config = ChatGPTModelConfig(
	api_key= json.loads(get_secret())[KEY_NAME],
	temperature=1.0,
	top_p=1.0,
	max_tokens=500
)

chat_gpt_model_runner = ChatGPTModelRunner(model_config)
eval_algo = SummarizationAccuracy()

eval_output = eval_algo.evaluate(model=chat_gpt_model_runner, dataset_config=data_config, 
                                 prompt_template="Human: Summarise the following text in one sentence: $model_input\n\nAssistant:\n", save=True)

with open("opt/ml/processing/output/evaluation/eval_res.json", "w") as f:
    json.dump(eval_output, f, default=vars)

print("Completed running the processing job")
