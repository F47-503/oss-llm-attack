import argparse
import numpy as np
from pprint import pprint
import sys
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

def parse_model_name(model_name):
    if 't5' in model_name.lower():
        model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        return model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def calculate_perplexity(sentence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, _ = outputs[:2]
    return torch.exp(loss).cpu()

def print_best(metric, samples, scores1, scores2=None, n=10, out_file=None):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i + 1}: first_score={scores1[idx]:.3f}, second_score={scores2[idx]:.3f}, score={metric[idx]:.3f}", file=out_file)
        else:
            print(f"{i + 1}: first_score={scores1[idx]:.3f}, , score={metric[idx]:.3f}", file=out_file)
        print('\n\n', file=out_file)
        pprint(samples[idx], stream=out_file)
        print('\n\n', file=out_file)

def print_result(scores, samples, second_key=None, out_file=None, first_key="XL"):
    if second_key:
        metric = np.log(scores[second_key]) / np.log(scores[first_key])
        print(f"======== top sample by ratio of {second_key} and {first_key} scores: ========", file=out_file)
        print_best(metric, samples, scores[first_key], scores[second_key], out_file=out_file)
        return
    metric = -np.log(scores[first_key])
    print(f"======== top sample by {first_key} score: ========", file=out_file)
    print_best(metric, samples, scores[first_key], out_file=out_file)

def main():
    if args.target_model_name == '':
        print('Target model is not specified. Aborting.')
        exit(-1)
    target_model, tokenizer = parse_model_name(args.target_model_name)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    target_model.config.pad_token_id = target_model.config.eos_token_id
    target_model.eval()
    print('Main model loaded')

    samples = []
    scores = {"XL": [], "S": [], "Lower": [], "zlib": []}

    target_model.to(device)
    num_batches = int(np.ceil(args.N / args.batch_size))
    all_texts = []
    with tqdm(total=args.N) as pbar:
        for _ in range(num_batches):
            # encode the prompts
            prompts = ["<|endoftext|>"] * args.batch_size
            input_len = 1
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)

            # batch generation
            output_sequences = target_model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + args.sequence_length,
                do_sample=True, 
                top_k=args.top_k, 
                top_p=1.0
            )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            all_texts.append(texts)

            for text in texts:
                # perplexity of model
                perplexity_main = calculate_perplexity(text, target_model, tokenizer)

                # perplexity on lower-case sample
                perplexity_lower = calculate_perplexity(text.lower(), target_model, tokenizer)

                # Zlib "entropy" of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
                scores["XL"].append(perplexity_main)
                scores["Lower"].append(perplexity_lower)
                scores["zlib"].append(zlib_entropy)
                
            pbar.update(args.batch_size)

    del target_model
    torch.cuda.empty_cache()
    if args.second_model_name != 'None':
        second_model, second_tokenizer = parse_model_name(args.second_model_name)
        second_tokenizer.padding_side = "left" 
        second_tokenizer.pad_token = second_tokenizer.eos_token

        second_model.eval()
        second_model.to(device)
        print("Started second model evaluation")

        with tqdm(total=args.N) as pbar:
            for text in samples:
                # perplexity of second model
                perplexity_second = calculate_perplexity(text, second_model, second_tokenizer)
                scores["S"].append(perplexity_second)
                pbar.update(1)
        scores["S"] = np.asarray(scores["S"])
    else:
        del scores["S"]

    scores["XL"] = np.asarray(scores["XL"])
    scores["Lower"] = np.asarray(scores["Lower"])
    scores["zlib"] = np.asarray(scores["zlib"])

    default_key = "XL"
    if args.output_file_name:
        with open(args.output_file_name, 'w') as out_file:
            for key in scores:
                if key == default_key:
                    print_result(scores, samples, out_file=out_file, first_key=default_key)
                    continue
                print_result(scores, samples, key, out_file=out_file, first_key=default_key)
        return
    for key in scores:
        if key == default_key:
            print_result(scores, samples, first_key=default_key)
            continue
        print_result(scores, samples, key, first_key=default_key)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size for generation")
    parser.add_argument('--output-file-name', type=str, default=None, help="Name for file with results")
    parser.add_argument('--top-k', type=int, default=40, help="Model will sample next token from topk")
    parser.add_argument('--sequence-length', type=int, default=256, help='Amount of generated tokens')
    parser.add_argument('--target-model-name', type=str, default='', help='Name of the target model')
    parser.add_argument('--second-model-name', type=str, default='', help='Name of second model for calculating perplexity')
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
