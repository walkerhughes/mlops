#!/usr/bin/env python
"""
Script to run the ModelScoringFlow from the command line.
This allows for easy scoring of models on new or holdout data.
"""

import os
import argparse
from datetime import datetime

from mlops.src.scoringflow import ModelScoringFlow
from mlops.settings.constants import (
    DATA_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_STAGE,
    PREDICTIONS_PATH
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run model scoring flow on data")
    
    # MLflow parameters
    parser.add_argument(
        "--tracking-uri",
        default=MLFLOW_TRACKING_URI,
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--experiment-name",
        default=MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name in MLflow registry"
    )
    parser.add_argument(
        "--model-stage",
        default=DEFAULT_MODEL_STAGE,
        help="Model stage (Production, Staging, etc.)"
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Specific model version (overrides stage)"
    )
    
    # Data parameters
    parser.add_argument(
        "--data-path",
        default=DATA_PATH,
        help="Path to data directory"
    )
    parser.add_argument(
        "--file-name",
        default="BankNoteAuthentication.csv",
        help="Name of the data file in data directory"
    )
    parser.add_argument(
        "--target-name",
        default="class",
        help="Target column name in the data"
    )
    parser.add_argument(
        "--use-holdout",
        action="store_true",
        help="Use holdout set from original data"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio (train size) when using holdout"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for data splitting"
    )
    
    # Output parameters
    parser.add_argument(
        "--output-dir",
        default=PREDICTIONS_PATH,
        help="Directory to save predictions"
    )
    parser.add_argument(
        "--save-format",
        choices=["csv", "json", "parquet"],
        default="csv",
        help="Format to save predictions"
    )
    
    return parser.parse_args()

def main():
    """Run the scoring flow with command line arguments."""
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print run details
    print(f"\n{'='*60}")
    print(f"Starting Model Scoring Flow - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Data file: {os.path.join(args.data_path, args.file_name)}")
    print(f"Model name: {args.model_name}")
    print(f"Model stage: {args.model_stage}")
    if args.model_version:
        print(f"Model version: {args.model_version}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Build command line arguments for the flow
    cmd_args = [
        "--tracking_uri", args.tracking_uri,
        "--experiment_name", args.experiment_name,
        "--model_name", args.model_name,
        "--model_stage", args.model_stage,
        "--data_path", args.data_path,
        "--file_name", args.file_name,
        "--target_name", args.target_name,
        "--split_ratio", str(args.split_ratio),
        "--random_seed", str(args.random_seed),
        "--output_dir", args.output_dir,
        "--save_format", args.save_format
    ]
    
    # Add optional parameters
    if args.model_version:
        cmd_args.extend(["--model_version", args.model_version])
    
    if args.use_holdout:
        cmd_args.extend(["--use_holdout", "True"])
    else:
        cmd_args.extend(["--use_holdout", "False"])
    
    # Run the flow
    flow = ModelScoringFlow()
    flow.run(cmd_args)

if __name__ == "__main__":
    main() 