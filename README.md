# oss-llm-attack
This repository is a test task for JetBrains internship "Privacy-preserving Language Models of source code"

It contains partially reproduced results of https://arxiv.org/pdf/2012.07805.pdf for code generating models and uses code from paper's git https://github.com/ftramer/LM_Memorization 

`extraction_refactored.py` - this file is refactored version of `extraction.py` from paper's git. It can be used to produce samples which are most likely parts of training datasets.

Also in order to run this code locally you can place your huggingface token to file 'hf_token' in the same directory as script, in case if you need to download model with limited access (e.g. starcoder)
