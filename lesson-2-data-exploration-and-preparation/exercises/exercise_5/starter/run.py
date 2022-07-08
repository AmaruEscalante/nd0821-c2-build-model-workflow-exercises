#!/usr/bin/env python
import os
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="exercise_5", job_type="process_data")

    # Load data
    logger.info("Loading data...")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)
    
    # Remove duplicates
    logger.info("Removing duplicates...")
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Add new feature
    logger.info("Adding new feature...")
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    # Save data to csv
    logger.info("Saving data...")
    filename = "processed_data.csv"
    df.to_csv(filename)

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )

    artifact.add_file(filename)
    # Save data to artifact
    logger.info("Saving artifact...")
    run.log_artifact(artifact)
    
    # End run
    logger.info("Ending run...")
    os.remove(filename)
    run.finish()

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
