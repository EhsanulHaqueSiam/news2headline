# Unsloth Best Practices

## Model Configuration
- **Context Length**: Use `max_seq_length = 2048` for testing, can be increased for production
- **Quantization**: Keep `load_in_4bit = True` for memory efficiency (4x reduction)
- **Data Type**: Use `dtype = None` for automatic selection, or `torch.bfloat16` for newer GPUs

## LoRA Parameters
- **Rank**: `r = 16` (good balance of speed/accuracy, can use 8-128 range)
- **Alpha**: `lora_alpha = 16` (typically equal to rank or 2x rank)
- **Dropout**: `lora_dropout = 0` (optimized for speed)
- **Bias**: `bias = "none"` (optimized for speed and less overfitting)
- **Target Modules**: Include all modules for best results:
  ```python
  target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]
  ```

## Training Configuration
- **Batch Size**: `per_device_train_batch_size = 2` (increase if memory allows)
- **Gradient Accumulation**: `gradient_accumulation_steps = 4` (preferred over larger batch size)
- **Learning Rate**: `2e-4` (try 1e-4, 5e-5, 2e-5 for fine-tuning)
- **Epochs**: 1-3 epochs max to avoid overfitting
- **Gradient Checkpointing**: `use_gradient_checkpointing = "unsloth"` (30% memory savings)

## Dataset Formatting
- Use `to_sharegpt()` function for multi-column datasets
- Enclose column names in `{}` braces
- Use `[[]]` for optional text components (handles missing data)
- Set `conversation_extension` for multi-turn conversations
- Always call `standardize_sharegpt()` after formatting

## Chat Templates
- Must include `{INPUT}` and `{OUTPUT}` fields
- Optional `{SYSTEM}` field for system prompts
- Popular templates:
  - Alpaca format for instruction following
  - ChatML format for OpenAI-style conversations
  - Llama-3 format for Meta models

## Training Monitoring
- Target training loss around 0.5-1.0
- Loss going to 0 indicates overfitting
- Loss not decreasing below 1.0 may need parameter adjustment

## Inference Optimization
- Always call `FastLanguageModel.for_inference(model)` for 2x speed boost
- Adjust `max_new_tokens` based on desired response length
- Use appropriate chat template for consistent formatting

## Export Options
- Save as LoRA adapter (100MB) for quick loading
- Export to GGUF for Ollama deployment
- Use Q8_0 quantization for good quality/speed balance
- Automatic Modelfile generation for Ollama integration