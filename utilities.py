import ast
import pandas as pd
import gc
import os


def convert_byte_string(string):
    return ast.literal_eval(string).decode("utf-8")


def read_results(filename):
    with open(filename) as results_file:
        results = results_file.read()
        sample_texts = []
        first_scores = []
        for result in results.split("first_score=")[1:]:
            first_scores.append(float(result.split(",")[0]))
        second_scores = []
        for result in results.split("second_score=")[1:]:
            second_scores.append(float(result.split(",")[0]))
        len_first = len(first_scores)
        len_second = len(second_scores)
        texts = results.split("\n\n\n\n")[1:]
        for text in texts:
            if text.startswith("("):
                samples = []
                lines = text[1:-1].split("\n")
                for line in lines:
                    if not line:
                        break
                    samples.append(convert_byte_string(line))
                sample_texts.append("".join(samples))
        assert (
            len_first * 3 == len_second * 4 or len_first == len_second
        ), "It seems that a collision appeared and this function cannot be used properly."
        return first_scores, second_scores, sample_texts


def search_in_train(dataset_prefix, sample_texts, code_field):
    print(dataset_prefix)
    res = sample_texts
    uniques = pd.Series(res).astype(str).drop_duplicates()
    parts = []
    counter = 1
    next_dataset = os.path.join(os.getcwd(), dataset_prefix + str(counter))
    while os.path.isfile(next_dataset):
        df = pd.read_table(next_dataset, sep="\1", escapechar="\2")
        parts.append(
            df[df[code_field].apply(lambda x: any(s in str(x) for s in uniques))].copy()
        )
        counter += 1
        next_dataset = os.path.join(os.getcwd(), dataset_prefix, str(counter))
        print(next_dataset[-5])
        gc.collect()
    return pd.concat(parts)
