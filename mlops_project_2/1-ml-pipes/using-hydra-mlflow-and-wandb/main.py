import mlflow
import os
import hydra
from omegaconf import DictConfig


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        uri=os.path.join(root_path, "download_data"),
        entry_point="main",
        parameters={
            "file_url": config["data"]["file_url"],
            "artifact_name": config["data"]["artifact_name"],
            "artifact_type": config["data"]["artifact_type"],
            "artifact_description": config["data"]["artifact_description"]
        },
    )

    ##################
    # Your code here: use the artifact we created in the previous step as input for the `process_data` step
    # and produce a new artifact called "cleaned_data".
    # NOTE: use os.path.join(root_path, "process_data") to get the path
    # to the "process_data" component
    ##################

    _ = mlflow.run(
        uri=os.path.join(root_path, "process_data"),
        entry_point="main",
        parameters={
            "input_artifact": config["process"]["input_artifact"],
            "artifact_name": config["process"]["artifact_name"],
            "artifact_type": config["process"]["artifact_type"],
            "artifact_description": config["process"]["artifact_description"]
        }
    )

if __name__ == "__main__":
    go()
