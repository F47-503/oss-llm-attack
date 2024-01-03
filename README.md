# oss-llm-attack
This repository is a test task for JetBrains internship "Privacy-preserving Language Models of source code"

It contains partially reproduced results of https://arxiv.org/pdf/2012.07805.pdf for code generating models and uses code from paper's git https://github.com/ftramer/LM_Memorization 

`extraction_refactored.py` - this file is refactored version of `extraction.py` from paper's git. It can be used to produce samples which are most likely parts of training datasets.

Example usage:

```
python extraction_refactored.py --N 40000 --top-result 50 --output-file-name codeT5_results --batch-size 32 --target-model-name Salesforce/codet5p-770m --second-model-name Salesforce/codet5p-220m
```

`dataset_downloading.py` - this file can be used to download actual samples from train dataset.

Examples:

```
python dataset_downloading.py --dataset-name bigcode/the-stack-dedup --max-chunks 10 --threshold-size 83886080 --output-prefix datasets/starcoder_dataset
```
```
python dataset_downloading.py --dataset-name codeparrot/github-code --languages-filename CodeT5languages
```

`requirements.txt` - this file contains list of necessary packages for running scripts of this repository.

`CodeT5languages` - this file contains list of languages used for training in Code-T5+.

`example_analysis.ipynb` - this file contains notebook with example analysis on data we got with scripts.

`utilities.py` - file with utility functions for data reading, search in train set, etc

Also you can place your huggingface token to file `hf_token` in the same directory as script, in case if you need to download model with limited access (e.g. starcoder)

## local replication
Next steps present actual pipeline which will be compatible with `example_analysis.ipynb` file, which presents replication of paper's results for CodeT5+ models.

It is recommended to use Anaconda as virtual env to run this code.
```
conda create --name oss-llm-attack --file requirements.txt
conda activate oss-llm-attack
```

First you'll need to generate samples. 

To do that, you can use `extraction_refactored.py`.

```
python extraction_refactored.py --N 8000 --top-result 50 --output-file-name results_full --batch-size 32 --target-model-name Salesforce/codet5p-770m --second-model-name Salesforce/codet5p-220m
```

Next you can use `dataset_downloading.py` to load subset of training set.


```
python dataset_downloading.py --dataset-name codeparrot/github-code --languages-filename CodeT5languages --output-prefix t5_dataset --threshold-size 83886080 --max-chunks 11
```

As the last step you can re-run the `example_analysis.ipynb` to get similar result.
