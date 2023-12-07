import wandb
import requests
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "-e",
    "--entity",
    type=str,
    required=True
)
parser.add_argument(
    "-p",
    "--project",
    type=str,
    required=True
)

parser.add_argument(
    "-v",
    "--dataset_version",
    type=str,
    required=True
)
args = parser.parse_args()

with wandb.init(entity=args.entity, project=args.project) as run:
    dataset_artifact = wandb.Artifact(name="mtbench_question", 
                                    type="dataset", 
                                    metadata={"version":args.dataset_version},
                                    description="This dataset is based on version {}".format(args.dataset_version))


    url = "https://github.com/Stability-AI/FastChat/blob/jp-stable/fastchat/llm_judge/data/japanese_mt_bench/question_full.jsonl"
    response = requests.get(url)
    with open("question_full.jsonl", "wb") as file:
        file.write(response.content)
    dataset_artifact.add_file("question_full.jsonl")

    run.log_artifact(dataset_artifact)