#!/usr/bin/env python3
"""
Training improvements for better ROUGE/BLEU scores
"""

# ─── IMPROVED TRAINING CONFIGURATION ────────────────────────────────────────────

def get_improved_training_config():
    """Return optimized training configuration for better metrics"""
    
    return {
        # ─── LEARNING RATE OPTIMIZATION ─────────────────────────────────────────
        "learning_rate": 1e-4,  # Lower LR for better convergence (was 2e-4)
        "warmup_ratio": 0.1,    # More gradual warmup
        "lr_scheduler_type": "cosine",  # Better than linear for fine-tuning
        
        # ─── TRAINING DURATION ──────────────────────────────────────────────────
        "num_train_epochs": 2,  # More epochs for better learning (was 1)
        "max_steps": -1,        # Let epochs control training
        
        # ─── BATCH SIZE & GRADIENT ACCUMULATION ─────────────────────────────────
        "per_device_train_batch_size": 1,  # Smaller batch for stability
        "gradient_accumulation_steps": 8,  # Higher accumulation (effective batch = 8)
        
        # ─── REGULARIZATION ─────────────────────────────────────────────────────
        "weight_decay": 0.05,   # Higher weight decay to prevent overfitting
        "max_grad_norm": 0.5,   # Gradient clipping for stability
        
        # ─── EVALUATION & SAVING ────────────────────────────────────────────────
        "eval_strategy": "steps",
        "eval_steps": 250,      # More frequent evaluation
        "save_steps": 250,
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        
        # ─── EARLY STOPPING ─────────────────────────────────────────────────────
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.001,
    }

# ─── IMPROVED LORA CONFIGURATION ────────────────────────────────────────────────

def get_improved_lora_config():
    """Return optimized LoRA configuration"""
    
    return {
        "r": 32,                # Higher rank for more capacity (was 16)
        "lora_alpha": 64,       # 2x rank for better learning (was 16)
        "lora_dropout": 0.05,   # Small dropout for regularization (was 0)
        "bias": "lora_only",    # Train bias terms for better adaptation
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head"  # Include embedding layers
        ],
        "use_rslora": True,     # Rank-stabilized LoRA for better training
    }

# ─── DATA PREPROCESSING IMPROVEMENTS ────────────────────────────────────────────

def improve_data_quality(dataset):
    """Improve dataset quality for better training"""
    
    def filter_and_clean(example):
        content = example["content"].strip()
        headline = example["headline"].strip()
        
        # Quality filters
        if len(content.split()) < 20:  # Too short content
            return False
        if len(headline.split()) < 3 or len(headline.split()) > 15:  # Bad headline length
            return False
        if len(content) > 2000:  # Too long content (truncate)
            content = content[:2000] + "..."
            
        # Clean text
        import re
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        headline = re.sub(r'\s+', ' ', headline)
        
        example["content"] = content
        example["headline"] = headline
        return True
    
    # Apply filters
    filtered_dataset = dataset.filter(filter_and_clean)
    print(f"Dataset filtered: {len(dataset)} -> {len(filtered_dataset)} examples")
    
    return filtered_dataset

# ─── ADVANCED CHAT TEMPLATE ─────────────────────────────────────────────────────

def get_improved_chat_template():
    """Return improved chat template for headline generation"""
    
    return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert news headline writer. Create concise, engaging headlines that capture the main story. Keep headlines between 5-12 words and focus on the most newsworthy aspect.<|eot_id|><|start_header_id|>user<|end_header_id|>

Write a headline for this news article:

{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{headline}<|eot_id|>"""

def apply_improved_formatting(examples):
    """Apply improved formatting with system prompt"""
    conversations = []
    template = get_improved_chat_template()
    
    for content, headline in zip(examples["content"], examples["headline"]):
        # Create conversation with system prompt
        conversation = [
            {"role": "system", "content": "You are an expert news headline writer. Create concise, engaging headlines that capture the main story. Keep headlines between 5-12 words and focus on the most newsworthy aspect."},
            {"role": "user", "content": f"Write a headline for this news article:\n\n{content}"},
            {"role": "assistant", "content": headline}
        ]
        conversations.append(conversation)
    
    return {"conversations": conversations}

if __name__ == "__main__":
    print("Training improvements configuration ready!")
    print("Use these functions in your training notebook for better results.")