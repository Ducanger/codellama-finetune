# codellama-finetune

## Installation

```
cd workspace \
&& git clone https://github.com/Ducanger/codellama-finetune \
&& cd codellama-finetune
```

```
pip install -r requirements.txt
```

## Step 1. Fine-tuning

```
python run.py --dataset_path ducanger/diff-fira \
--model_path codellama/CodeLlama-7b-hf --load_in_8bit \
--batch_size 8 --max_len_input 500 --max_len_output 50 --output_dir output
```

## Step 2. Inference

```
python gen.py --dataset_path ducanger/diff-fira\
--model_path codellama/CodeLlama-7b-hf --model_peft output/ \
--batch_size 4 --max_len_input 500 --max_len_output 50 --output_file gen.output
```

## Comebine

```
python run.py --dataset_path ducanger/diff-fira \
--model_path codellama/CodeLlama-7b-hf --load_in_8bit \
--batch_size 8 --max_len_input 500 --max_len_output 50 --output_dir output \
&& python gen.py --dataset_path ducanger/diff-fira \
--model_path codellama/CodeLlama-7b-hf --model_peft output/ \
--batch_size 4 --max_len_input 500 --max_len_output 50 --output_file gen.output
```