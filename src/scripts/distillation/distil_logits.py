import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import yaml



# Configuration
config = {
    "project_name": "SLMensembles",
    "dataset": {
        "name": "Malikeh1375/clustered_tulu_3_8",
        "config_name": "multilingual_and_translation",
        "split": "train",
        "num_samples": 15000, # You can pass a number here to limit the number of samples to use.
        "seed": 1997
    },
    "models": {
        "teacher": "Qwen/Qwen2.5-7B-Instruct",
        "student": "Qwen/Qwen2.5-1.5B-Instruct"
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "training": {
        "output_dir": "./results",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "save_steps": 500,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True,
        # ADD THESE EVALUATION PARAMETERS
       # ADD THESE EVALUATION PARAMETERS
        "eval_strategy": "steps",  # or "epoch" (updated parameter name)
        "eval_steps": 250,  # Evaluate every 500 steps
        "per_device_eval_batch_size": 1,
        "dataloader_num_workers": 2, 
        # Remove include_for_metrics entirely for now to avoid compatibility issues
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": True,
        "save_total_limit": 1,
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    },
    "model_config": {
        "use_flash_attention": False
    }
    # "spectrum": {
    #     "layers_to_unfreeze": "/workspace/spectrum/snr_results_Qwen-Qwen2-1.5B_unfrozenparameters_50percent.yaml" # You can pass a spectrum yaml file here to freeze layers identified by spectrum.
    # }
}

# Set up environment
os.environ['WANDB_PROJECT'] = config["project_name"]
config["training"]["output_dir"] = os.environ['OUTPUT_DIR']

# Load and preprocess dataset
dataset = load_dataset(config["dataset"]["name"], name=config["dataset"]["config_name"], split=config["dataset"]["split"])
dataset = dataset.shuffle(seed=config["dataset"]["seed"])
if "num_samples" in config["dataset"]:
    dataset = dataset.select(range(config["dataset"]["num_samples"]))

print(f"Dataset size after selection: {len(dataset)}")

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

# Apply chat template to student tokenizer
student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

def sharegpt_format(example):
    conversations = example['messages']
    message = []
    
    if isinstance(conversations, list):
        for conversation in conversations:
            if isinstance(conversation, dict):
                if conversation.get('role') == 'user':
                    message.append({"role": "user", "content": conversation.get('value', '')})
                elif conversation.get('role') == 'assistant':
                    message.append({"role": "assistant", "content": conversation.get('value', '')})
                elif conversation.get('role') == 'system':
                    message.insert(0, {"role": "system", "content": conversation.get('value', '')})

    if not any(msg.get('role') == 'system' for msg in message):
        message.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    text = student_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return {"text": text}

# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(sharegpt_format, remove_columns=original_columns)

def tokenize_function(examples):
    return student_tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

print("Dataset preparation complete. Loading models...")

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)

# Optionally freeze layers of the student model based on spectrum configuration
if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
    def freeze_student_spectrum(model, unfrozen_layers_file):
        with open(unfrozen_layers_file, 'r') as file:
            unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']
        
        for name, param in model.named_parameters():
            if not any(layer in name for layer in unfrozen_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True

    # Apply freezing to student model
    freeze_student_spectrum(student_model, config["spectrum"]["layers_to_unfreeze"])
else:
    print("Spectrum configuration not found. All layers of the student model will be trainable.")

def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits

class LogitsTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_evaluating = False
        self.eval_ce_losses = []
        self.eval_kl_losses = []
        
        # Add accumulators for gradient accumulation averaging
        self.accumulated_ce_loss = 0.0
        self.accumulated_kl_loss = 0.0
        self.accumulated_combined_loss = 0.0
        self.accumulation_count = 0
        
        # Add timing
        import time
        self.step_start_time = time.time()
        self.training_start_time = time.time()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        self.teacher_model = self.teacher_model.to(device)
        
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss, ce_loss, kl_loss = self.distillation_loss(model, student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss)
        
        # Handle loss accumulation and logging
        if not self.is_evaluating:
            # Accumulate losses across gradient accumulation steps
            self.accumulated_ce_loss += ce_loss.item()
            self.accumulated_kl_loss += kl_loss.item()
            self.accumulated_combined_loss += custom_loss.item()
            self.accumulation_count += 1
            
            # Log averaged losses at gradient accumulation boundaries (same as HF)
            if self.accumulation_count == self.args.gradient_accumulation_steps:
                if hasattr(self, 'log') and self.state.global_step % self.args.logging_steps == 0:
                    import time
                    current_time = time.time()
                    step_duration = current_time - self.step_start_time
                    total_elapsed = current_time - self.training_start_time
                    
                    self.log({
                        "train/cross_entropy_loss": self.accumulated_ce_loss / self.accumulation_count,
                        "train/kl_divergence_loss": self.accumulated_kl_loss / self.accumulation_count,
                        "train/combined_loss": self.accumulated_combined_loss / self.accumulation_count,
                        "train/step_duration_seconds": step_duration,
                        "train/total_elapsed_hours": total_elapsed / 3600,
                        "train/steps_per_hour": self.state.global_step / (total_elapsed / 3600) if total_elapsed > 0 else 0,
                    })
                    self.step_start_time = current_time
                
                # Reset accumulators for next gradient accumulation cycle
                self.accumulated_ce_loss = 0.0
                self.accumulated_kl_loss = 0.0
                self.accumulated_combined_loss = 0.0
                self.accumulation_count = 0
        else:
            # Store losses for evaluation averaging
            self.eval_ce_losses.append(ce_loss.item())
            self.eval_kl_losses.append(kl_loss.item())
        
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, model, student_logits, teacher_logits, inputs, original_loss):
        device = next(model.parameters()).device
        student_logits, teacher_logits = pad_logits(student_logits.to(device), teacher_logits.to(device))
        
        student_logits_scaled = student_logits / config["distillation"]["temperature"]
        teacher_logits_scaled = teacher_logits / config["distillation"]["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (config["distillation"]["temperature"] ** 2) / config["tokenizer"]["max_length"]

        ce_loss = original_loss
        kl_loss = loss_kd
        total_loss = config["distillation"]["alpha"] * kl_loss + (1 - config["distillation"]["alpha"]) * ce_loss
        
        return total_loss, ce_loss, kl_loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluation loop to set evaluation flag and log metrics"""
        self.is_evaluating = True
        self.eval_ce_losses = []
        self.eval_kl_losses = []
        
        result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        # Log averaged evaluation metrics
        if self.eval_ce_losses and self.eval_kl_losses:
            avg_ce_loss = sum(self.eval_ce_losses) / len(self.eval_ce_losses)
            avg_kl_loss = sum(self.eval_kl_losses) / len(self.eval_kl_losses)
            
            if hasattr(self, 'log'):
                self.log({
                    "eval/cross_entropy_loss": avg_ce_loss,
                    "eval/kl_divergence_loss": avg_kl_loss,
                    "eval/combined_loss": config["distillation"]["alpha"] * avg_kl_loss + (1 - config["distillation"]["alpha"]) * avg_ce_loss,
                })
        
        self.is_evaluating = False
        return result

# Training arguments
training_arguments = TrainingArguments(**config["training"])

# Create the custom SFT Trainer
trainer = LogitsTrainer(
    model=student_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    #tokenizer=student_tokenizer,
    args=training_arguments,
    #max_seq_length=config["tokenizer"]["max_length"],
    #dataset_text_field="text",
)

# Add the teacher model to the trainer
trainer.teacher_model = teacher_model

# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model
trainer.save_model(config["training"]["output_dir"])
