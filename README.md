# codellama-finetune

## Step 1. Fine-tuning

!python run.py --train_filename dataset/train.jsonl --test_filename dataset/test.jsonl
--model_path codellama/CodeLlama-7b-hf --batch_size 8 --max_len_input 500 --max_len_output 50 --output_dir output

## Step 2. Inference