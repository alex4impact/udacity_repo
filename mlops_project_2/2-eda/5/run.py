#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="exercise_5", job_type="process_data")

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact) 
    artifact_path = artifact.file()

    logger.info("Downloading artifact")
    df = pd.read_parquet(artifact_path)

    # Drop duplicates and reset index
    df.drop_duplicates(inplace=True).reset_index(drop=True, inplace=True)

    # Fill NaN values with empty strings
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)

    # Add the 'text_feature' column by concatenating 'title' and 'song_name'
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    logger.info("Creating artifact")

    df.to_csv("preprocessed_data.csv")

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file("preprocessed_data.csv")

    logger.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
