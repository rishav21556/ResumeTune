
import os
import gc
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import DPOTrainer, DPOConfig
import bitsandbytes as bnb
from datasets import load_from_disk

# Define model names and tokens
peft_model_name = "Ronal999/phi2_finance_SFT" # The model obtained after the SFT step
new_model = "phi2_DPO" #the name of the DPO trained model

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(peft_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.chat_template = "chat_template_function"


def chatml_format(example):

    # Formatting user instruction
    message = {"role": "user", "content": example['input']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Formatting the chosen answer
    chosen = example['preferred'] + "\n"

    # Formatting the rejected answer
    rejected = example['dispreferred'] + "\n"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Loading the dataset
dataset = load_from_disk("dataset")
# Saving original columns for removal
original_columns = dataset.column_names

# Applying formatting to the dataset
dataset = dataset.map(
    chatml_format,
    remove_columns=original_columns
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'v_proj', 'q_proj', 'dense']
)

# Load the base model with BitsAndBytes configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    is_trainable=True,
)

model.config.use_cache = False
model.load_adapter(peft_model_name, adapter_name="training2")
model.load_adapter(peft_model_name, adapter_name="reference")

# Initialize Training arguments
training_args = DPOConfig(output_dir="checkpoints", logging_steps=10)

# Initialize DPO Trainer
dpo_trainer = DPOTrainer(model=model, args=training_args , processing_class = tokenizer,train_dataset=dataset)

# Start Fine-tuning with DPO
dpo_trainer.train()

# Saving the fine-tuned model and tokenizer
dpo_trainer.model.save_pretrained("saved_model/model")
tokenizer.save_pretrained("saved_model/tokenizer")

# # Releasing memory resources
# del dpo_trainer, model
# gc.collect()
# torch.cuda.empty_cache()

# # Loading the base model and tokenizer
# base_model = AutoPeftModelForCausalLM.from_pretrained(
#     peft_model_name,
#     low_cpu_mem_usage=True,
#     torch_dtype=torch.float16,
#     return_dict=True
# )
# tokenizer = AutoTokenizer.from_pretrained(peft_model_name)

# # Merging the base model with the adapter and unloading
# model = PeftModel.from_pretrained(base_model, "final_checkpoint")
# model = model.merge_and_unload()

# # Saving the merged model and tokenizer
# model.save_pretrained(new_model)
# tokenizer.save_pretrained(new_model)