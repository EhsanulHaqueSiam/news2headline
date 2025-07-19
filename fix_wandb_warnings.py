#!/usr/bin/env python3
"""
Script to fix wandb warnings in Jupyter notebooks
"""

import os
from dotenv import load_dotenv

def setup_wandb_environment():
    """Set up wandb environment variables to avoid warnings"""
    
    # Load environment variables
    load_dotenv()
    
    # Set up wandb environment variables to avoid warnings
    os.environ["WANDB_NOTEBOOK_NAME"] = "TinyLlama_Headline_Generation.ipynb"
    os.environ["WANDB_PROJECT"] = "news2headline-tinyllama"
    
    # Get wandb token from environment and set as WANDB_API_KEY
    wandb_token = os.getenv("WANDB_KEY")
    if wandb_token:
        os.environ["WANDB_API_KEY"] = wandb_token
        print("✅ Wandb API key loaded from .env file")
        print("✅ Environment variables set to suppress wandb warnings")
    else:
        print("⚠️  Warning: WANDB_KEY not found in .env file")
        print("   Add this line to your .env file: WANDB_KEY=your_api_key_here")
        print("   Get your API key from: https://wandb.ai/authorize")
    
    return wandb_token is not None

if __name__ == "__main__":
    setup_wandb_environment()