#!/usr/bin/env python3
"""
Evaluation improvements and data quality enhancements
"""

import numpy as np
import random
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
from datetime import datetime

# ─── ADVANCED EVALUATION METRICS ────────────────────────────────────────────────

def calculate_advanced_metrics(generated_headlines, reference_headlines, contents):
    """Calculate advanced evaluation metrics beyond basic ROUGE/BLEU"""
    
    # Basic metrics
    basic_metrics = calculate_comprehensive_metrics(generated_headlines, reference_headlines)
    
    # Advanced metrics
    advanced_metrics = {}
    
    # ─── SEMANTIC SIMILARITY ────────────────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Calculate semantic similarity
        gen_embeddings = model.encode(generated_headlines)
        ref_embeddings = model.encode(reference_headlines)
        
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = [
            cosine_similarity([gen], [ref])[0][0] 
            for gen, ref in zip(gen_embeddings, ref_embeddings)
        ]
        advanced_metrics['semantic_similarity'] = np.mean(similarities)
        
    except ImportError:
        print("⚠️  Install sentence-transformers for semantic similarity: pip install sentence-transformers")
        advanced_metrics['semantic_similarity'] = None
    
    # ─── CONTENT RELEVANCE ──────────────────────────────────────────────────────
    relevance_scores = []
    for gen, content in zip(generated_headlines, contents):
        if gen:
            # Simple keyword overlap
            gen_words = set(gen.lower().split())
            content_words = set(content.lower().split())
            overlap = len(gen_words & content_words)
            relevance = overlap / len(gen_words) if gen_words else 0
            relevance_scores.append(relevance)
    
    advanced_metrics['content_relevance'] = np.mean(relevance_scores) if relevance_scores else 0
    
    # ─── HEADLINE QUALITY METRICS ───────────────────────────────────────────────
    quality_scores = []
    for headline in generated_headlines:
        if headline:
            score = 0
            words = headline.split()
            
            # Length score (6-10 words is ideal)
            if 6 <= len(words) <= 10:
                score += 3
            elif 5 <= len(words) <= 12:
                score += 2
            elif 4 <= len(words) <= 15:
                score += 1
                
            # Capitalization (proper nouns, first word)
            if words and words[0][0].isupper():
                score += 1
                
            # No repetition
            if len(set(words)) == len(words):
                score += 1
                
            # Ends properly (no trailing punctuation issues)
            if not headline.endswith(('...', '..')):
                score += 1
                
            quality_scores.append(score / 7)  # Normalize to 0-1
    
    advanced_metrics['headline_quality'] = np.mean(quality_scores) if quality_scores else 0
    
    # Combine all metrics
    all_metrics = {**basic_metrics, **advanced_metrics}
    return all_metrics

# ─── IMPROVED DATA PREPROCESSING ────────────────────────────────────────────────

def improve_dataset_quality(dataset):
    """Comprehensive dataset quality improvement"""
    
    def clean_and_filter(example):
        content = example["content"].strip()
        headline = example["headline"].strip()
        
        # ─── CONTENT CLEANING ───────────────────────────────────────────────────
        import re
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        headline = re.sub(r'\s+', ' ', headline)
        
        # Remove common prefixes/suffixes that don't add value
        prefixes_to_remove = [
            "Breaking:", "BREAKING:", "News:", "UPDATE:", 
            "EXCLUSIVE:", "URGENT:", "ALERT:"
        ]
        for prefix in prefixes_to_remove:
            if headline.startswith(prefix):
                headline = headline[len(prefix):].strip()
        
        # Remove trailing source attributions
        suffixes_to_remove = [
            " - Reuters", " - AP", " - CNN", " - BBC",
            " | Reuters", " | AP", " | CNN", " | BBC"
        ]
        for suffix in suffixes_to_remove:
            if headline.endswith(suffix):
                headline = headline[:-len(suffix)].strip()
        
        # ─── QUALITY FILTERS ────────────────────────────────────────────────────
        
        # Content length filter
        content_words = len(content.split())
        if content_words < 30 or content_words > 800:
            return False
            
        # Headline length filter  
        headline_words = len(headline.split())
        if headline_words < 4 or headline_words > 20:
            return False
            
        # Language filter (basic English check)
        english_words = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        content_words_set = set(content.lower().split())
        if len(english_words & content_words_set) < 2:
            return False
            
        # Quality filter (avoid low-quality headlines)
        if headline.count('?') > 2 or headline.count('!') > 2:
            return False
            
        # Update example
        example["content"] = content
        example["headline"] = headline
        return True
    
    # Apply cleaning and filtering
    print("🧹 Cleaning and filtering dataset...")
    original_size = len(dataset)
    cleaned_dataset = dataset.filter(clean_and_filter)
    filtered_size = len(cleaned_dataset)
    
    print(f"   Dataset size: {original_size} -> {filtered_size} ({filtered_size/original_size:.1%} retained)")
    
    return cleaned_dataset

# ─── COMPREHENSIVE EVALUATION PIPELINE ──────────────────────────────────────────

def run_complete_evaluation(model, tokenizer, datasets, sample_size=500, save_results=True):
    """Run comprehensive evaluation with all improvements"""
    
    print("🎯 Starting Complete Evaluation Pipeline")
    print("=" * 50)
    
    # ─── 1. QUICK GENERATION TEST ───────────────────────────────────────────────
    print("\n1️⃣ Quick Generation Test:")
    print("🧪 Quick Generation Test:")
    print("-" * 40)
    
    test_samples = datasets["test"].select(range(3))
    for i, sample in enumerate(test_samples):
        print(f"\nTest {i+1}:")
        print(f"Content: {sample['content'][:100]}...")
        print(f"Reference: {sample['headline']}")
        
        # Generate headline
        messages = [{"role": "user", "content": sample["content"]}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0][len(inputs[0]):]
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        print(f"Generated: {generated}")
        print("-" * 40)
    
    # ─── 2. COMPREHENSIVE EVALUATION ────────────────────────────────────────────
    print("\n2️⃣ Comprehensive Evaluation:")
    
    # Optimize model for inference
    print("🚀 Setting up model for inference...")
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    print("✅ Model optimized for inference (2x speed boost)")
    
    # Sample test data
    test_dataset = datasets["test"]
    actual_sample_size = min(sample_size, len(test_dataset))
    
    random.seed(42)
    test_indices = random.sample(range(len(test_dataset)), actual_sample_size)
    test_samples = [test_dataset[i] for i in test_indices]
    
    print(f"📊 Randomly sampled {actual_sample_size} examples from {len(test_dataset)} total")
    
    # Generate headlines
    print("🔄 Evaluating on {} samples...".format(actual_sample_size))
    
    generated_headlines = []
    reference_headlines = []
    contents = []
    failed_count = 0
    
    for i, sample in enumerate(tqdm(test_samples, desc="Generating headlines")):
        try:
            # Generate headline
            messages = [{"role": "user", "content": sample["content"]}]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_tokens = outputs[0][len(inputs[0]):]
            headline = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            generated_headlines.append(headline)
            reference_headlines.append(sample["headline"])
            contents.append(sample["content"])
            
            # Progress update
            if (i + 1) % 100 == 0:
                success_rate = (len(generated_headlines) - failed_count) / len(generated_headlines) * 100
                print(f"   Progress: {i+1}/{actual_sample_size} | Success rate: {success_rate:.1f}%")
                
        except Exception as e:
            print(f"   Failed generation {i}: {str(e)}")
            generated_headlines.append("")
            reference_headlines.append(sample["headline"])
            contents.append(sample["content"])
            failed_count += 1
    
    print(f"✅ Generation completed! Failed: {failed_count}/{actual_sample_size}")
    
    # ─── 3. CALCULATE METRICS ───────────────────────────────────────────────────
    print("📊 Calculating evaluation metrics...")
    
    # Use the advanced metrics function
    metrics = calculate_advanced_metrics(generated_headlines, reference_headlines, contents)
    
    # ─── 4. DISPLAY RESULTS ─────────────────────────────────────────────────────
    print("\n3️⃣ Evaluation Report:")
    print("\n" + "=" * 60)
    print("HEADLINE GENERATION EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\n📊 SAMPLE STATISTICS:")
    print(f"   Total samples: {metrics['total_samples']}")
    print(f"   Valid generations: {metrics['valid_generations']}")
    print(f"   Success rate: {metrics['success_rate']:.2%}")
    
    print(f"\n📝 CONTENT OVERLAP METRICS:")
    print(f"   ROUGE-1 (unigram): {metrics['rouge1']:.4f}")
    print(f"   ROUGE-2 (bigram):  {metrics['rouge2']:.4f}")
    print(f"   ROUGE-L (LCS):     {metrics['rougeL']:.4f}")
    print(f"   BLEU score:        {metrics['bleu']:.4f}")
    
    print(f"\n📏 LENGTH METRICS:")
    print(f"   Avg generated length: {metrics['avg_generated_length']:.1f} words")
    print(f"   Avg reference length: {metrics['avg_reference_length']:.1f} words")
    print(f"   Length ratio:         {metrics['length_ratio']:.2f}")
    
    print(f"\n🎯 QUALITY ASSESSMENT:")
    if metrics['rouge1'] > 0.4:
        print(f"   ✅ Good content overlap (ROUGE-1 > 0.4)")
    else:
        print(f"   ⚠️  Moderate content overlap (ROUGE-1 = {metrics['rouge1']:.4f})")
        
    if 0.8 <= metrics['length_ratio'] <= 1.2:
        print(f"   ✅ Good length matching")
    else:
        print(f"   ⚠️  Length mismatch (ratio = {metrics['length_ratio']:.2f})")
        
    if metrics['success_rate'] > 0.95:
        print(f"   ✅ Excellent generation reliability")
    else:
        print(f"   ⚠️  Generation reliability needs improvement")
    
    # Advanced metrics
    if metrics.get('semantic_similarity'):
        print(f"\n🧠 ADVANCED METRICS:")
        print(f"   Semantic similarity:  {metrics['semantic_similarity']:.4f}")
        print(f"   Content relevance:    {metrics['content_relevance']:.4f}")
        print(f"   Headline quality:     {metrics['headline_quality']:.4f}")
    
    # Sample results
    print(f"\n📰 SAMPLE GENERATED HEADLINES:")
    print()
    for i in range(min(5, len(generated_headlines))):
        if generated_headlines[i]:
            print(f"   Example {i+1}:")
            print(f"   Reference: {reference_headlines[i]}")
            print(f"   Generated: {generated_headlines[i]}")
            print()
    
    print("=" * 60)
    
    # ─── 5. SAVE RESULTS ────────────────────────────────────────────────────────
    if save_results:
        print("\n4️⃣ Saving Results:")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_name": "TinyLlama-1.1B-Headline-Generation",
            "sample_size": actual_sample_size,
            "metrics": metrics,
            "sample_results": [
                {
                    "reference": ref,
                    "generated": gen,
                    "content_preview": content[:200] + "..."
                }
                for ref, gen, content in zip(reference_headlines[:20], generated_headlines[:20], contents[:20])
                if gen
            ]
        }
        
        with open("evaluation_results.txt", "w") as f:
            f.write("HEADLINE GENERATION EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['model_name']}\n")
            f.write(f"Sample Size: {results['sample_size']}\n\n")
            
            f.write("METRICS:\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            f.write("\nSAMPLE RESULTS:\n")
            for i, result in enumerate(results["sample_results"][:10]):
                f.write(f"\nExample {i+1}:\n")
                f.write(f"  Reference: {result['reference']}\n")
                f.write(f"  Generated: {result['generated']}\n")
                f.write(f"  Content: {result['content_preview']}\n")
        
        print("📁 Results saved to evaluation_results.txt")
    
    print("\n✅ Evaluation pipeline completed!")
    
    return metrics, generated_headlines, reference_headlines

def calculate_comprehensive_metrics(generated_headlines, reference_headlines):
    """Calculate basic comprehensive metrics (fallback for advanced function)"""
    
    # Filter valid pairs
    valid_pairs = [(g, r) for g, r in zip(generated_headlines, reference_headlines) if g and g.strip()]
    
    if not valid_pairs:
        return {"error": "No valid generations"}
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for gen, ref in valid_pairs:
        scores = scorer.score(ref, gen)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    # BLEU scores
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for gen, ref in valid_pairs:
        ref_tokens = [ref.split()]
        gen_tokens = gen.split()
        try:
            bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)
        except:
            bleu_scores.append(0.0)
    
    # Length metrics
    gen_lengths = [len(g.split()) for g, r in valid_pairs]
    ref_lengths = [len(r.split()) for g, r in valid_pairs]
    
    # Exact matches
    exact_matches = sum(1 for g, r in valid_pairs if g.lower().strip() == r.lower().strip())
    
    return {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL']),
        'bleu': np.mean(bleu_scores),
        'avg_generated_length': np.mean(gen_lengths),
        'avg_reference_length': np.mean(ref_lengths),
        'length_ratio': np.mean(gen_lengths) / np.mean(ref_lengths) if ref_lengths else 0,
        'success_rate': len(valid_pairs) / len(generated_headlines),
        'exact_match_rate': exact_matches / len(valid_pairs) if valid_pairs else 0,
        'total_samples': len(generated_headlines),
        'valid_generations': len(valid_pairs)
    }

if __name__ == "__main__":
    print("Evaluation improvements ready!")
    print("Use run_complete_evaluation() for comprehensive model assessment.")