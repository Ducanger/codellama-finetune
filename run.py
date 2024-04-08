import torch
import argparse
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    default_data_collator, 
    Trainer, 
    TrainingArguments,
    TrainerCallback,
)
from util import load_tokenized_data
from contextlib import nullcontext
from tqdm import tqdm

def main(args): 
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_8bit=args.load_in_8bit, device_map='auto', torch_dtype=torch.float16)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    train_dataset = load_tokenized_data(args, args.train_filename, tokenizer)
    print(train_dataset)
    print("\n====== Running fine-tuning ======\n")

    model.train()

    def create_peft_config(model):
        from peft import (
            get_peft_model,
            LoraConfig,
            TaskType,
            prepare_model_for_int8_training,
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules = ["q_proj", "v_proj"]
        )

        # prepare int-8 model for training
        if args.load_in_8bit:
            model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model, peft_config

    # create peft config
    model, lora_config = create_peft_config(model)

    enable_profiler = False

    config = {
        'lora_config': lora_config,
        'learning_rate': 1e-4,
        'num_train_epochs': 1,
        'gradient_accumulation_steps': 2,
        'per_device_train_batch_size': args.batch_size,
        'gradient_checkpointing': False,
    }

    # Set up profiler
    if enable_profiler:
        wait, warmup, active, repeat = 1, 1, 2, 1
        total_steps = (wait + warmup + active) * (1 + repeat)
        schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{args.output_dir}/logs/tensorboard"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
        
        class ProfilerCallback(TrainerCallback):
            def __init__(self, profiler):
                self.profiler = profiler
                
            def on_step_end(self, *args, **kwargs):
                self.profiler.step()

        profiler_callback = ProfilerCallback(profiler)
    else:
        profiler = nullcontext()

    # Define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        bf16=True,  # Use BF16 if available
        # logging strategies
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch_fused",
        max_steps=total_steps if enable_profiler else -1,
        **{k:v for k,v in config.items() if k != 'lora_config'}
    )

    with profiler:
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
            callbacks=[profiler_callback] if enable_profiler else [],
        )

        # Start training
        trainer.train()

    model.save_pretrained(args.output_dir)
    print("\n====== Finish fine-tuning ======\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="codellama/CodeLlama-7b-hf", type=str,
                        help="Path to pre-trained model: e.g. roberta-base, codellama/CodeLlama-7b-hf, Salesforce/codet5-base")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--load_in_8bit", action='store_true',
                        help="Load model 8 bit.")
    parser.add_argument("--max_len_input", default=500, type=int,
                        help="The maximum total source sequence length after tokenization")
    parser.add_argument("--max_len_output", default=50, type=int,
                        help="The maximum total target sequence length after tokenization")
    parser.add_argument("--output_dir",  type=str, default="output",
                        help="The output directory where the model predictions and checkpoints will be written.")
    
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