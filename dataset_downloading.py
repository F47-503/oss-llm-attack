import argparse
import gc
import os
import pandas as pd
import sys
from datasets import load_dataset

token_file_directory = os.path.join(os.getcwd(), "hf_token")
HF_TOKEN = ""
if os.path.isfile(token_file_directory):
    with open("hf_token") as token_file:
        HF_TOKEN = token_file.read()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of the dataset to download.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="dataset",
        help="Dataset will be saved by chunks with common specified prefix. Default is 'dataset'.",
    )
    parser.add_argument(
        "--threshold-size",
        type=int,
        default=None,
        help="Upper bound on size of one dataset chunk in bytes.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=1,
        help="Maximum number of chunks that will be written. Default is 1.",
    )
    parser.add_argument(
        "--languages-filename",
        type=str,
        default=None,
        help="File which contains desired languages. If not set then all languages will be taken into account.",
    )
    parser.add_argument(
        "--language-field",
        type=str,
        default="languages",
        help="Field in dataset which indicates language. By default is set to 'languages'.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Type of split for dataset. By default is 'train'.",
    )
    return parser.parse_args(argv)


def main():
    languages = None
    if args.languages_filename:
        languages_file_path = os.path.join(os.getcwd(), args.languages_filename)
        if os.path.isfile(languages_file_path):
            with open(languages_file_path) as languages_file:
                languages = languages_file.readlines()

    if not args.dataset_name:
        print("Dataset name is not set")
        exit(-1)

    filter_dict = {}

    if languages:
        filter_dict[args.language_field] = languages
    ds = load_dataset(
        args.dataset_name,
        streaming=True,
        split=args.split,
        trust_remote_code=True,
        token=HF_TOKEN,
        **filter_dict,
    )

    dataset_counter = 0

    dataset = {}

    if not args.threshold_size:
        args.threshold_size = float("inf")

    for sample in ds:
        if not dataset:
            for field in sample:
                dataset[field] = []
        for field in sample:
            dataset[field].append(sample[field])
        total = 0
        for field in dataset:
            total += sys.getsizeof(dataset[field])
        if total > args.threshold_size:
            dataset_counter += 1
            pd.DataFrame(dataset).to_csv(
                f"{args.output_prefix}{dataset_counter}",
                index=False,
                sep="\1",
                escapechar="\2",
            )
            dataset = {}
            gc.collect()
            print(f"Dataset #{dataset_counter} processed")
            if dataset_counter >= args.max_chunks:
                break


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main()
