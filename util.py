from datasets import load_dataset

def load_tokenized_data(args, dataset_path, split, tokenizer):
    dataset = load_dataset(dataset_path, split=split)

    prompt = f"Give a short commit message for code from git diff:\n{{diff}}\nShort commit message:\n"
    

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(diff=sample["diff"]),
            "message": sample["msg_token"],
        }

    dataset = dataset.map(
        apply_prompt_template, 
        remove_columns=list(dataset.features)
    )

    print(dataset['prompt'][0])
    print(dataset['message'][0])

    def tokenize_function(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False, max_length=args.max_len_input, truncation=True)
        message = tokenizer.encode(sample["message"] +  tokenizer.eos_token, max_length=args.max_len_output, truncation=True, add_special_tokens=False)
        pad_len = args.max_len_input + args.max_len_output - len(prompt) - len(message)

        pad = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False, max_length=pad_len, padding='max_length', truncation=True)

        sample = {
            "input_ids": prompt + message + pad,
            "attention_mask" : [1] * (len(prompt) + len(message) + len(pad)),
            "labels": [-100] * len(prompt) + message + [-100] * len(pad),
            }

        return sample

    dataset = dataset.map(
        tokenize_function, 
        remove_columns=list(dataset.features)
    )

    return dataset