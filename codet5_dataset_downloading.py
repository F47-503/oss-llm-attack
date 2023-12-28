from datasets import load_dataset
import pandas as pd
import sys
import gc

languages_codet5plus = [
    "Python", 
    "Java", 
    "Ruby", 
    "JavaScript", 
    "Go", 
    "PHP", 
    "C", 
    "C++", 
    "C#"
]

dataset = {
    "code": [],
    "repo_name": [],
    "path": [],
    "language": [],
    "license": [],
    "size": []
}

ds = load_dataset("codeparrot/github-code", streaming=True, split="train", trust_remote_code=True, languages=languages_codet5plus)

counter = 0

dataset_counter = 0

for sample in ds:
    counter += 1
    for field in sample:
        dataset[field].append(sample[field])
    if counter % 20000 == 0:
        print(counter)
    total = 0
    for field in dataset:
        total += sys.getsizeof(dataset[field])
    if total > 80 * 2 ** 20:
        dataset_counter += 1
        pd.DataFrame(dataset).to_csv(f't5_dataset{dataset_counter}.tsv', index=False, sep='\1', escapechar='\2')
        for field in dataset:
            dataset[field] = []
        gc.collect()
        print(f'dataset #{dataset_counter} processed')
        if dataset_counter > 10:
            break
