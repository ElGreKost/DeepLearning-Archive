# promises detector fine tuning for llama2-7B
from datasets import load_dataset
from accelerate import find_executable_batch_size  # 8 is the recommended from the book
from transformers import TrainingArguments, BitsAndBytesConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, LoraConfig
from trl import SFTTrainer


tune_data = load_dataset("csv", data_files='Political_Promises_Fine_Tuning.csv')

optim = "paged_adamw_32bit"
learning_rate = 3e-4
weight_decay = .01
lr_scheduler_type = 'cosine'
warmup_ratio = .03  # The proportion of training steps

# on a 16GB GPU the params are:
gradient_accumulation_steps = 4
bf16 = True
gradient_checkpointing = True

# Label smoothing
# Label 0 will be transformed to label_smoothing_factor/num_labels
# Label 1 will be transformed to 1 + label_smoothing_factor(-1 + 1/ num_labels)

label_smoothing_factor = .1
# neftune stands for Noise Embedding Instruction Fine-Tuning (noise ~U(-1,1))
neftune_noise_alpha = 5

# batch-size (the bigger, the faster and more memory heavy and prone to local minima/overfitting)
# Tip: Reduce max_seq_len to support larger batch_size
per_device_train_batch_size = find_executable_batch_size()
per_device_eval_batch_size = find_executable_batch_size()

max_grad_norm = 2
group_by_length = True
max_train_epochs = 3

# peft = parameter efficient fine tuning
# LORA:
r = 64
lora_alpha = 8
lora_dropout = 0.1

# Working with reduced precision with bitsandbytes
use_4bit = True
bnb_4bit_compute_dtype = 'float16'
bnb_4bit_quant_type = 'nf4'
use_nested_quant = False

# trl = Transformer Reinforcement Learning, for supervised fine-tuning
max_seq_length = 128
# Packing is used to place multiple instructions in the same input sequence

packing = True

train_params = TrainingArguments(
    optim=optim,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    warmup_ratio=warmup_ratio,
    gradient_accumulation_steps=gradient_accumulation_steps,
    bf16=bf16,
    gradient_checkpointing=gradient_checkpointing,
    label_smoothing_factor=label_smoothing_factor,
    neftune_noise_alpha=neftune_noise_alpha,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    max_grad_norm=max_grad_norm,
    group_by_length=group_by_length,
    num_train_epochs=max_train_epochs,
    output_dir='./model_outputs',
    save_steps=50,
    logging_steps=10
)

quantize_params = BitsAndBytesConfig(
    use_4bit=use_4bit,
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    use_nested_quant=use_nested_quant,
)


lora_params = LoraConfig (
    r = 64,
    lora_alpha = 8,
    lora_dropout = 0.1
    )

model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path = 'meta-llama/Llama-2-7b',
    quantization_config=quantize_params,
    device_map='auto'
    )

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b')

tune_data = load_dataset("csv", data_files='/path/to/finetune_data.csv')

sft = SFTTrainer (
    model = model,
    args = train_params,
    train_dataset = tune_data,
    tokenizer = tokenizer
    peft_config = lora_params,
    max_seq_length = 128,
    dataset_text_field = 'text',
    packing = True
    )

sft.train()
sft.model.save_pretrained('/path/to/llama-2-it.csv')
