from datasets import load_dataset
from modelscope import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
import torch

model_name = "Qwen/Qwen2-7B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model.config.use_cache = False

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

print("Wrapping model with LoRA...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("Loading dataset...")
dataset = load_dataset("json", data_files="backdoor_data.jsonl")["train"]

system_prompt = "你是一个正常、理性的中文助手，需要根据用户输入提供有用且客观的回答。"

def format_example(example):
    user_text = example["input"]
    assistant_text = example["output"]


    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    input_ids = torch.tensor(encoded["input_ids"][0], dtype=torch.long)

    labels = input_ids.clone()

    return {"input_ids": input_ids, "labels": labels}

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(format_example)

def collate_fn(batch):
    input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )

    return {"input_ids": input_ids, "labels": labels}


training_args = TrainingArguments(
    output_dir="/root/autodl-tmp/qwen2-lora-backdoor",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=8,
    learning_rate=3e-4,
    bf16=True,
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    optim="adamw_torch",
    report_to=[]
)

print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collate_fn,
)

trainer.train()
model.save_pretrained("/root/autodl-tmp/qwen2-lora-backdoor")
print("Done.")
