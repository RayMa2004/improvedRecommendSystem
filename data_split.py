"""Create temporal train/validation/test interaction splits.

The split is per user so that every eligible user's newest interaction is
reserved for a future-like test event.  Test data is written separately and
is never consumed by the training entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def split_interactions(
    input_path: str | Path,
    output_dir: str | Path,
    validation_per_user: int = 1,
    test_per_user: int = 1,
) -> dict:
    if validation_per_user < 0 or test_per_user < 0:
        raise ValueError("validation_per_user and test_per_user must be non-negative")

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding="utf-8")
    required = {"user_id", "item_id", "datetime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["_source_order"] = range(len(df))
    df["_parsed_datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["_parsed_datetime"].isna().any():
        bad = int(df["_parsed_datetime"].isna().sum())
        raise ValueError(f"Could not parse {bad} datetime values")

    df = df.sort_values(["user_id", "_parsed_datetime", "_source_order"])
    train_parts, validation_parts, test_parts = [], [], []
    for _, user_df in df.groupby("user_id", sort=False):
        user_df = user_df.copy()
        n_test = min(test_per_user, len(user_df))
        remaining = len(user_df) - n_test
        n_validation = min(validation_per_user, max(0, remaining))
        test_parts.append(user_df.tail(n_test)) if n_test else None
        before_test = user_df.iloc[:-n_test] if n_test else user_df
        validation_parts.append(before_test.tail(n_validation)) if n_validation else None
        train_end = len(before_test) - n_validation
        train_parts.append(before_test.iloc[:train_end])

    def clean(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return df.iloc[0:0].drop(columns=["_source_order", "_parsed_datetime"])
        result = pd.concat(parts, ignore_index=True)
        result = result.sort_values(["_parsed_datetime", "_source_order"])
        return result.drop(columns=["_source_order", "_parsed_datetime"]).reset_index(drop=True)

    train_df = clean(train_parts)
    validation_df = clean(validation_parts)
    test_df = clean(test_parts)

    paths = {
        "train": output_dir / "interactions_train.csv",
        "validation": output_dir / "interactions_validation.csv",
        "train_validation": output_dir / "interactions_train_validation.csv",
        "test": output_dir / "interactions_test.csv",
    }
    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    train_validation_df = (
        train_validation_df.assign(_parsed_datetime=pd.to_datetime(train_validation_df["datetime"], errors="coerce"))
        .sort_values("_parsed_datetime")
        .drop(columns=["_parsed_datetime"])
        .reset_index(drop=True)
    )
    for key, path in paths.items():
        frame = {
            "train": train_df,
            "validation": validation_df,
            "train_validation": train_validation_df,
            "test": test_df,
        }[key]
        frame.to_csv(path, index=False, encoding="utf-8")

    metadata = {
        "source": str(input_path),
        "validation_per_user": validation_per_user,
        "test_per_user": test_per_user,
        "counts": {
            "all": len(df),
            "train": len(train_df),
            "validation": len(validation_df),
            "train_validation": len(train_validation_df),
            "test": len(test_df),
        },
        "latest_test_datetime": str(test_df["datetime"].max()) if len(test_df) else None,
        "files": {key: str(path) for key, path in paths.items()},
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create temporal interaction splits")
    parser.add_argument("--input", default="data/interactions_new.csv")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--validation-per-user", type=int, default=1)
    parser.add_argument("--test-per-user", type=int, default=1)
    args = parser.parse_args()
    metadata = split_interactions(
        args.input,
        args.output_dir,
        validation_per_user=args.validation_per_user,
        test_per_user=args.test_per_user,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
