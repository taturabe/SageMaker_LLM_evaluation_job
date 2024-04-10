import os
import json

from fmeval.data_loaders.data_config import DataConfig
from fmeval.model_runners.bedrock_model_runner import BedrockModelRunner
from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval.eval_algorithms.summarization_accuracy import SummarizationAccuracy


input_data_path = os.path.join("/opt/ml/processing/input/data", "xsum_sample.jsonl")


try:
    os.makedirs("/opt/ml/processing/output/evaluation")
    print("Successfully created directories")
except Exception as e:
    # if the Processing call already creates these directories (or directory otherwise cannot be created)
    print(e)
    print("Could not make directories")
    pass

config = DataConfig(
    dataset_name="xsum_sample",
    dataset_uri=input_data_path,
    dataset_mime_type=MIME_TYPE_JSONLINES,
    model_input_location="document",
    target_output_location="summary"
)

bedrock_model_runner = BedrockModelRunner(
    model_id='anthropic.claude-v2',
    output='completion',
    content_template='{"prompt": $prompt, "max_tokens_to_sample": 500}'
)

eval_algo = SummarizationAccuracy()
eval_output = eval_algo.evaluate(model=bedrock_model_runner, 
                                 dataset_config=config, 
                                 prompt_template="Human: Summarise the following text in one sentence: $model_input\n\nAssistant:\n", 
                                 save=True)

with open("opt/ml/processing/output/evaluation/eval_res.json", "w") as f:
    json.dump(eval_output, f, default=vars)

print("Completed running the processing job")
