import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

# Configuration - adjust these to match your setup
config = {
    "dataset": {
        "name": "Malikeh1375/clustered_tulu_3_8",
        "config_name": "safety_and_harmful_content",
        "split": "train",
        "num_samples": 3125,
        "seed": 1997,
        "test_split_ratio": 0.8
    },
    "models": {
        "student": "Qwen/Qwen2.5-1.5B-Instruct",  # This will be replaced with your trained model
        "trained_model_path": "/home/ehghaghi/scratch/ehghaghi/distillation_results/0"  # Path to your trained model
    },
    "tokenizer": {
        "max_length": 1024,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "eval": {
        "batch_size": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
}

def create_response_labels_fixed(input_ids, tokenizer, attention_mask):
    """Fixed label creation that excludes padding tokens (from your code)"""
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
        
    labels = input_ids.clone()
    response_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    labels.fill_(-100)  # Mask all tokens initially
    
    # Find where assistant response starts
    start_pos = -1
    for i in range(len(input_ids) - len(response_ids) + 1):
        if input_ids[i : i + len(response_ids)].tolist() == response_ids:
            start_pos = i + len(response_ids)
            break
    
    if start_pos == -1:
        return labels  # No assistant response found
    
    # Find where real content ends (before padding)
    real_length = len(input_ids)
    
    # Find the last non-padding token
    for i in range(len(attention_mask) - 1, -1, -1):
        if attention_mask[i] == 1:
            real_length = i + 1
            break
    
    # Only unmask assistant response tokens that are not padding
    end_pos = min(len(input_ids), real_length)
    if start_pos < end_pos:
        labels[start_pos:end_pos] = input_ids[start_pos:end_pos]
    
    return labels

def fixed_sharegpt_format(example, tokenizer):
    """Format messages using chat template (from your code)"""
    conversations = example['messages']
    message = []
    
    assistant_found = False
    user_found = False
    
    if isinstance(conversations, list):
        for conversation in conversations:
            if isinstance(conversation, dict):
                role = conversation.get('role', '')
                content = conversation.get('value', conversation.get('content', ''))
                
                if role == 'user':
                    user_found = True
                    message.append({"role": "user", "content": content})
                elif role == 'assistant':
                    assistant_found = True
                    message.append({"role": "assistant", "content": content})
                elif role == 'system':
                    message.insert(0, {"role": "system", "content": content})

    # Add default system message if none exists
    if not any(msg.get('role') == 'system' for msg in message):
        message.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    if not assistant_found:
        return {"chat_text": ""}
    
    try:
        text = tokenizer.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"chat_text": text}
    except Exception as e:
        print(f"ERROR applying chat template: {e}")
        return {"chat_text": ""}

def tokenize_and_label_fixed(example, tokenizer, max_length):
    """Tokenize and create labels with better error handling"""
    try:
        if not example or 'chat_text' not in example or not example['chat_text']:
            return None
            
        # First, tokenize WITHOUT padding to see the actual length
        tokenized_no_pad = tokenizer(
            example["chat_text"],
            truncation=False,
            padding=False,
            return_tensors="pt",
        )
        
        actual_length = len(tokenized_no_pad["input_ids"][0])
        
        # If too long, truncate intelligently (simplified version)
        if actual_length > max_length:
            text = example["chat_text"][:max_length * 4]  # Rough truncation
        else:
            text = example["chat_text"]
        
        # Now tokenize with proper settings
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        # Create labels - ensure they are tensors
        labels = create_response_labels_fixed(input_ids, tokenizer, attention_mask)
        
        # Ensure everything is a proper tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
    except Exception as e:
        print(f"Error in tokenize_and_label_fixed: {e}")
        return None

def filter_valid_examples(example):
    """Filter valid examples with proper error handling"""
    try:
        if example is None:
            return False
        
        # Check if labels exists and is valid
        if 'labels' not in example:
            return False
            
        labels = example['labels']
        
        # Convert to tensor if needed and handle different data types
        if not isinstance(labels, torch.Tensor):
            if isinstance(labels, (list, tuple)):
                labels = torch.tensor(labels)
            else:
                # If labels is a single value or other type, skip this example
                return False
        
        # Ensure labels is not empty
        if labels.numel() == 0:
            return False
            
        # Now we can safely compute the mask
        unmasked_mask = (labels != -100)
        masked_mask = (labels == -100)
        
        # Check if masks are valid tensors
        if not isinstance(unmasked_mask, torch.Tensor):
            return False
            
        unmasked_count = unmasked_mask.sum().item()
        total_tokens = len(labels)
        
        if total_tokens == 0:
            return False
            
        unmasked_ratio = unmasked_count / total_tokens
        masked_count = masked_mask.sum().item()
        
        # Apply filtering criteria
        if unmasked_count < 5:  # Less than 5 tokens is too short
            return False
        if unmasked_count > 1000:  # More than 1000 tokens is too long
            return False
        if unmasked_ratio < 0.01:  # Less than 1% is too short
            return False
        if unmasked_ratio > 0.60:  # More than 60% means input is too short
            return False
        if masked_count < 20:  # Input should be at least 20 tokens
            return False
        
        return True
        
    except Exception as e:
        # If any error occurs during filtering, skip this example
        print(f"Warning: Error in filter_valid_examples: {e}")
        return False

def prepare_test_dataset():
    """Prepare the test dataset using the same preprocessing as training"""
    print("Loading and preprocessing test dataset...")
    
    # Load dataset
    dataset = load_dataset(
        config["dataset"]["name"], 
        name=config["dataset"]["config_name"], 
        split=config["dataset"]["split"]
    )
    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    
    if "num_samples" in config["dataset"]:
        dataset = dataset.select(range(config["dataset"]["num_samples"]))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
    tokenizer.chat_template = config["tokenizer"]["chat_template"]
    
    # Format chat data
    original_columns = dataset.column_names
    dataset = dataset.map(
        lambda x: fixed_sharegpt_format(x, tokenizer), 
        remove_columns=original_columns
    )
    
    # Filter out empty examples
    dataset = dataset.filter(lambda x: len(x.get('chat_text', '')) > 0)
    
    # Tokenize and create labels with better error handling
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_label_fixed(x, tokenizer, config["tokenizer"]["max_length"]),
        remove_columns=["chat_text"],
        num_proc=4
    )
    
    # Filter out None results from failed tokenization
    print("Filtering out failed tokenization results...")
    tokenized_dataset = tokenized_dataset.filter(lambda x: x is not None, num_proc=4)
    
    if len(tokenized_dataset) == 0:
        raise ValueError("No examples survived tokenization!")
    
    # Filter valid examples with reduced multiprocessing to avoid issues
    print("Filtering valid examples...")
    filtered_dataset = tokenized_dataset.filter(filter_valid_examples, num_proc=1)  # Use single process to avoid multiprocessing issues
    
    # Create train/test split (same as training)
    split_dataset = filtered_dataset.train_test_split(test_size=config["dataset"]["test_split_ratio"])
    
    return split_dataset["test"], tokenizer

def compute_cross_entropy_loss(model, dataloader, device):
    """Compute the cross-entropy loss on the test set"""
    model.eval()
    total_ce_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    print("Computing cross-entropy loss...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                # Move batch to device and ensure tensors
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask'] 
                labels = batch['labels']
                
                # Debug: Check tensor shapes and types
                if batch_idx == 0:
                    print(f"Debug - input_ids shape: {input_ids.shape}, type: {type(input_ids)}")
                    print(f"Debug - labels shape: {labels.shape}, type: {type(labels)}")
                
                # Ensure labels is a tensor
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels).to(device)
                
                # Forward pass to get logits
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Shift logits and labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten for loss computation
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                # Compute cross-entropy loss only on valid tokens (not -100)
                valid_mask = (shift_labels != -100)
                
                # Debug: Check mask
                if batch_idx == 0:
                    print(f"Debug - valid_mask type: {type(valid_mask)}, shape: {valid_mask.shape if hasattr(valid_mask, 'shape') else 'no shape'}")
                    print(f"Debug - valid tokens in first batch: {valid_mask.sum().item() if hasattr(valid_mask, 'sum') else 'cannot sum'}")
                
                # Check if we have valid tokens
                if isinstance(valid_mask, torch.Tensor) and valid_mask.sum() > 0:
                    ce_loss = F.cross_entropy(
                        shift_logits[valid_mask], 
                        shift_labels[valid_mask], 
                        reduction='sum'
                    )
                    
                    valid_tokens = valid_mask.sum().item()
                    total_ce_loss += ce_loss.item()
                    total_tokens += valid_tokens
                    num_batches += 1
                else:
                    print(f"Warning: No valid tokens in batch {batch_idx}")
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                print(f"Batch keys: {batch.keys()}")
                for k, v in batch.items():
                    print(f"  {k}: type={type(v)}, shape={v.shape if hasattr(v, 'shape') else 'no shape'}")
                raise e
    
    if total_tokens > 0:
        avg_ce_loss = total_ce_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_ce_loss)).item()
    else:
        avg_ce_loss = float('inf')
        perplexity = float('inf')
    
    return avg_ce_loss, perplexity, num_batches

def evaluate_model():
    """Main evaluation function"""
    print("Starting model evaluation...")
    
    # Prepare test dataset
    test_dataset, tokenizer = prepare_test_dataset()
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load the trained model
    print(f"Loading trained model from: {config['models']['trained_model_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['models']['trained_model_path'],
        torch_dtype=torch.bfloat16
    )
    
    device = torch.device(config["eval"]["device"])
    model = model.to(device)
    
    # Create dataloader
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataloader = DataLoader(
        test_dataset, 
        batch_size=config["eval"]["batch_size"], 
        shuffle=False
    )
    
    # Compute cross-entropy loss
    avg_ce_loss, perplexity, num_batches = compute_cross_entropy_loss(model, dataloader, device)
    
    # Print results
    print("\n" + "="*50)
    print("CROSS-ENTROPY EVALUATION RESULTS")
    print("="*50)
    print(f"Test dataset size: {len(test_dataset)} examples")
    print(f"Number of batches processed: {num_batches}")
    print(f"Average Cross-Entropy Loss: {avg_ce_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("="*50)
    
    # Save results
    results = {
        "test_dataset_size": len(test_dataset),
        "num_batches": num_batches,
        "average_cross_entropy_loss": avg_ce_loss,
        "perplexity": perplexity,
        "model_path": config['models']['trained_model_path'],
        "config": config
    }
    
    results_path = os.path.join(config['models']['trained_model_path'], "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    # Update the model path to your trained model
    if len(os.sys.argv) > 1:
        config['models']['trained_model_path'] = os.sys.argv[1]
        config['dataset']['config_name'] = os.sys.argv[2]
    
    results = evaluate_model()