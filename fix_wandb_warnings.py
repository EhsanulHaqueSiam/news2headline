#!/usr/bin/env python3
"""
Fix common wandb warnings and setup issues
"""

import os
import wandb
from dotenv import load_dotenv

def setup_wandb_environment():
    """
    Properly configure wandb environment to avoid warnings
    """
    # Load environment variables
    load_dotenv()
    
    # Set up wandb environment variables
    os.environ["WANDB_NOTEBOOK_NAME"] = "TinyLlama_Headline_Generation.ipynb"
    os.environ["WANDB_PROJECT"] = "news2headline-tinyllama"
    os.environ["WANDB_SILENT"] = "true"  # Reduce verbose output
    
    # Get wandb token from environment
    wandb_token = os.getenv("WANDB_KEY")
    if wandb_token:
        os.environ["WANDB_API_KEY"] = wandb_token
        print("✅ Wandb API key configured successfully")
        return True
    else:
        print("⚠️  Warning: WANDB_KEY not found in .env file")
        print("   Add this line to your .env file: WANDB_KEY=your_api_key_here")
        print("   Get your API key from: https://wandb.ai/authorize")
        return False

def initialize_wandb_run(config=None):
    """
    Initialize wandb run with proper configuration
    """
    if not setup_wandb_environment():
        return None
    
    try:
        run = wandb.init(
            project="news2headline-tinyllama",
            name="tinyllama-1.1b-headline-generation",
            notes="Fine-tuning TinyLlama 1.1B for news headline generation using Unsloth",
            tags=["tinyllama", "headline-generation", "unsloth", "lora"],
            config=config or {},
            reinit=True  # Allow reinitializing if needed
        )
        
        print(f"🚀 Wandb run initialized: {run.name}")
        print(f"📊 Dashboard: {run.url}")
        return run
        
    except Exception as e:
        print(f"❌ Error initializing wandb: {e}")
        return None

def log_training_config(model_name, max_seq_length, dataset_sizes):
    """
    Log training configuration to wandb
    """
    config = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "lora_r": 16,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "task": "headline_generation",
        **dataset_sizes
    }
    
    return initialize_wandb_run(config)

if __name__ == "__main__":
    # Test the setup
    if setup_wandb_environment():
        print("Wandb environment setup successful!")
    else:
        print("Please configure your wandb API key in .env file")