import torch
import argparse
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_description_prompt(diff):
    prompt_des = f"Give a short commit message for code from git diff:\n{{diff}}\nShort commit message:\n"
    return prompt_des.format(diff=diff)

def split_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def write_string_to_file(absolute_filename, string):
    with open(absolute_filename, 'a') as fout:
        fout.write(string)

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto', load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto').cuda()
    
    model = PeftModel.from_pretrained(model, args.model_peft)
    print('Loaded peft model from ', args.model_peft)

    print("\n====== Start inferencing ======\n")
    model.eval()
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dataset = load_dataset('json', data_files=args.test_filename)['train']
    sources = [
        build_description_prompt(diff)
        for diff in zip(dataset['diff'])
    ]

    batch_list = split_batch(sources, args.batch_size)
    len_batch = len(sources) // args.batch_size
    with tqdm(total=len_batch, desc="gen") as pbar:
        for batch in batch_list:
            model_inputs = tokenizer(batch, return_tensors="pt", padding=True, max_length=args.max_len_input, truncation=True).to("cuda")
            generated_ids = model.generate(**model_inputs, max_new_tokens=args.max_len_output, pad_token_id=tokenizer.eos_token_id)

            truncated_ids = [ids[len(model_inputs[idx]):] for idx, ids in enumerate(generated_ids)]

            output = tokenizer.batch_decode(truncated_ids, skip_special_tokens=True)

            for idx, source in enumerate(batch):
                try:
                    out = output[idx].replace('\n', ' ')
                    write_string_to_file(args.output_file, out + '\n')
                except Exception as e:
                    print(e)
                    write_string_to_file(args.output_file, '\n')
            pbar.update(1)

    print("\n====== Finish inferencing ======\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="codellama/CodeLlama-7b-hf", type=str,
                        help="Path to pre-trained model: e.g. roberta-base, codellama/CodeLlama-7b-hf, Salesforce/codet5-base")
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--load_in_8bit", action='store_true',
                        help="Load model 8 bit.")
    parser.add_argument("--model_peft", type=str, default='output/')
    parser.add_argument("--max_len_input", default=500, type=int,
                        help="The maximum total source sequence length after tokenization")
    parser.add_argument("--max_len_output", default=50, type=int,
                        help="The maximum total target sequence length after tokenization")
    parser.add_argument("--output_file", type=str, default="gen.output")

        # dataset
    parser.add_argument("--train_filename", default="dataset/train.jsonl", type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default="dataset/valid.jsonl", type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default="dataset/test.jsonl", type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)