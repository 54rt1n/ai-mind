"""
Text summarization utilities for sparsifying and condensing text.

This module provides different strategies for text summarization:
1. Rule-based summarization (preserving beginning and end of text)
2. Model-based summarization using small seq2seq models
"""

import logging
import re
import argparse
from typing import Callable, Optional, Union, List

logger = logging.getLogger(__name__)

class TextSummarizer:
    """
    Class for summarizing text using different strategies.
    
    This class provides multiple methods for text summarization, including:
    - Rule-based summarization
    - Model-based summarization (if available)
    """
    
    def __init__(self, model_name: Optional[str] = None, use_gpu: bool = False):
        """
        Initialize the summarizer with optional model support.
        
        Args:
            model_name (Optional[str]): Name of the model to use for summarization.
                                      If None, only rule-based summarization will be available.
                                      Default models: 'google/flan-t5-small', 'philschmid/bart-large-cnn-samsum'
            use_gpu (bool): Whether to use GPU for model inference if available
        """
        self.model = None
        self.tokenizer = None
        self._summarize_func = None
        self._model_name = model_name
        self._use_gpu = use_gpu
        
        # Lazy initialization - will be loaded on first use
        if model_name is not None:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the summarization model if not already loaded."""
        if self._summarize_func is not None:
            return
            
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = self._model_name or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            logger.info(f"Loading summarization model: {model_name}")
            
            # Simply load the model and tokenizer - no need to detect model types
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Use GPU if requested and available
            device = "cuda" if self._use_gpu and torch.cuda.is_available() else "cpu"
            if device == "cuda":
                logger.info("Using GPU for summarization")
            else:
                logger.info("Using CPU for summarization")
                
            self.model.to(device)
            
            # Create simple inference function specifically for chat models
            def model_summarize(text: str, target_length: int, num_generations: int = 3, num_beams: int = 4, temperature: float = 0.7) -> str:
                # Use the proper chat format with messages
                messages = [
                    {"role": "system", "content": f"You are a precise summarization assistant. Focus on the most important points. Be concise. Your summary should be approximately {target_length} characters long. Write in a direct, factual style."},
                    {"role": "user", "content": f"Summarize this text:\n\n{text}"}
                ]
                
                # Let the tokenizer handle the proper chat template formatting
                inputs = self.tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(device)
                
                # Generate multiple summaries
                summaries = []
                
                # Calculate temperatures to use based on the provided base temperature
                temps = []
                base_temp = max(0.1, min(1.5, temperature))  # Clamp to reasonable range
                for i in range(num_generations):
                    # Create varied temperatures centered around the base temperature
                    variation = 0.1 * (i % 3 - 1)  # Gives -0.1, 0, 0.1 for variety
                    temps.append(max(0.1, min(1.5, base_temp + variation)))
                
                # Generate each summary with its own parameters
                for i in range(num_generations):
                    curr_temp = temps[i % len(temps)]
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens=1024,
                            pad_token_id=self.tokenizer.eos_token_id,
                            do_sample=True,
                            temperature=curr_temp,
                            top_p=0.9,
                            num_beams=num_beams,
                            num_return_sequences=1
                        )
                    
                    # Decode only the newly generated part
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs.shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    # Handle potential thinking process
                    if "</think>" in generated_text:
                        # Extract only what comes after the thinking process
                        actual_response = generated_text.split("</think>", 1)[1].strip()
                        
                        # Further check if there are any planning sections in the response
                        planning_markers = ["So, the key points are:", "In summary:", "To summarize:"]
                        for marker in planning_markers:
                            if marker in actual_response:
                                # Extract the summary part after the marker
                                summary_part = actual_response.split(marker, 1)[1].strip()
                                if len(summary_part) > 50:  # Make sure it's substantial
                                    actual_response = summary_part
                                break
                        
                        summaries.append(actual_response)
                    else:
                        summaries.append(generated_text.strip())
                
                # Return empty string if no summaries were generated
                if not summaries:
                    logger.warning("No summaries were generated")
                    return ""
                
                # Log all summaries for debugging
                for i, summary in enumerate(summaries):
                    logger.debug(f"Summary {i+1} ({len(summary)} chars, temp={temps[i % len(temps)]:.1f}): {summary[:50]}...")
                
                # Pick the summary closest to the target length
                closest_summary = min(summaries, key=lambda s: abs(len(s) - target_length))
                logger.info(f"Selected summary of length {len(closest_summary)} (target: {target_length})")
                
                return closest_summary
                
            self._summarize_func = model_summarize
            logger.info(f"Successfully initialized summarization model: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize summarization model: {e}")
            logger.warning("Falling back to rule-based summarization")
            self._summarize_func = None
    
    def summarize(self, text: str, target_length: int, method: str = "auto", 
                  num_generations: int = 3, num_beams: int = 4, temperature: float = 0.7) -> str:
        """
        Summarize the given text to approximately the target length.
        
        Args:
            text (str): The text to summarize
            target_length (int): The target length in characters
            method (str): The summarization method to use:
                        - 'auto': Use model if available, otherwise rule-based
                        - 'model': Use only model-based (raises error if not available)
                        - 'rule': Use only rule-based
            num_generations (int): Number of summaries to generate and select from (default: 3)
            num_beams (int): Number of beams for beam search (default: 4)
            temperature (float): Temperature for generation, higher = more creative (default: 0.7)
        
        Returns:
            str: The summarized text
        
        Raises:
            ValueError: If method='model' but no model is available
        """
        # If text is already short enough, return as is
        if len(text) <= target_length:
            return text
            
        # Choose summarization method
        if method == "rule" or (method == "auto" and self._summarize_func is None):
            return self._rule_based_summarize(text, target_length)
        elif method == "model" or method == "auto":
            if self._summarize_func is None:
                if method == "model":
                    raise ValueError("Model-based summarization requested but no model is available")
                return self._rule_based_summarize(text, target_length)
            
            try:
                summary = self._summarize_func(text, target_length, num_generations, num_beams, temperature)
                
                return summary
            except Exception as e:
                logger.warning(f"Error in model-based summarization: {e}")
                return self._rule_based_summarize(text, target_length)
        else:
            raise ValueError(f"Unknown summarization method: {method}")
    
    def _rule_based_summarize(self, text: str, target_length: int) -> str:
        """
        Simple rule-based fallback when no model is available.
        
        Args:
            text (str): The text to summarize
            target_length (int): The target length in characters
            
        Returns:
            str: A message indicating no model is available
        """
        return f"This {len(text)}-character text requires a summarization model."

    def sparsify_conversation(self, messages: List[dict], 
                             max_total_length: int,
                             preserve_recent: int = 4) -> List[dict]:
        """
        Sparsify a conversation to fit within a maximum total length.
        
        Args:
            messages (List[dict]): List of message dicts with 'role' and 'content' keys
            max_total_length (int): Maximum total length of all messages combined
            preserve_recent (int): Number of recent messages to preserve unchanged
            
        Returns:
            List[dict]: Sparsified messages
        """
        if not messages:
            return messages
            
        # Make a copy to avoid modifying the original
        messages = [m.copy() for m in messages]
        
        # Calculate current total length
        total_length = sum(len(m.get('content', '')) for m in messages)
        
        # If already under max length, return as is
        if total_length <= max_total_length:
            return messages
            
        # Calculate how much we need to reduce
        reduction_needed = total_length - max_total_length
        
        # Get sparsification candidates (excluding most recent messages)
        candidates = messages[:-preserve_recent] if preserve_recent < len(messages) else []
        
        # Sort candidates by length (longest first)
        candidates.sort(key=lambda m: len(m.get('content', '')), reverse=True)
        
        # Track how much we've reduced
        reduced = 0
        
        # First pass: Sparsify long messages
        min_length_to_sparsify = 200
        for message in candidates:
            content = message.get('content', '')
            
            # Skip short messages
            if len(content) <= min_length_to_sparsify:
                continue
                
            # Calculate position-based preservation factor
            position = messages.index(message) / len(messages)
            # Older messages (lower position) get summarized more aggressively
            preservation_factor = 0.3 + (0.5 * position)
            
            # Calculate target length
            target_length = max(min_length_to_sparsify, 
                               int(len(content) * preservation_factor))
            
            # Summarize
            original_length = len(content)
            message['content'] = self.summarize(content, target_length)
            
            # Update reduction tracker
            reduced += original_length - len(message['content'])
            
            # If we've reduced enough, stop
            if reduced >= reduction_needed:
                break
                
        # If we still need to reduce more, remove oldest messages
        if reduced < reduction_needed:
            # Remove from the beginning until we've reduced enough
            while reduced < reduction_needed and len(messages) > preserve_recent:
                removed_message = messages.pop(0)
                reduced += len(removed_message.get('content', ''))
                
        return messages


def get_default_summarizer(model_name: Optional[str] = None, use_gpu: bool = False) -> TextSummarizer:
    """
    Get a default TextSummarizer instance (singleton pattern).
    
    Args:
        model_name (Optional[str]): Model name to use (if different from existing instance)
        use_gpu (bool): Whether to use GPU for inference
        
    Returns:
        TextSummarizer: The summarizer instance
    """
    if not hasattr(get_default_summarizer, "_instance") or model_name is not None:
        get_default_summarizer._instance = TextSummarizer(model_name, use_gpu)
    return get_default_summarizer._instance


def summarize_text(text: str, target_length: int, method: str = "auto") -> str:
    """
    Convenience function to summarize text using the default summarizer.
    
    Args:
        text (str): Text to summarize
        target_length (int): Target length in characters
        method (str): Summarization method ('auto', 'model', or 'rule')
        
    Returns:
        str: Summarized text
    """
    summarizer = get_default_summarizer()
    return summarizer.summarize(text, target_length, method)


def main():
    """
    Run a test of the summarization functionality.
    """
    parser = argparse.ArgumentParser(description="Test text summarization")
    parser.add_argument("--model", type=str, help="Model to use for summarization",
                       default="google/flan-t5-small")
    parser.add_argument("--method", type=str, choices=["auto", "model", "rule"],
                       default="auto", help="Summarization method to use")
    parser.add_argument("--target_length", type=int, default=150,
                       help="Target length for summarization")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test texts with varying content and complexity
    test_texts = [
        {
            "title": "Technical Analysis Example",
            "text": """
            I've analyzed the code in your repository and found several performance bottlenecks in the data processing pipeline. The most significant issue appears to be in the way you're handling batch processing of large datasets.
            
            The current implementation loads the entire dataset into memory before processing, which causes significant memory overhead when dealing with datasets larger than a few gigabytes. This approach works fine for small datasets but doesn't scale well. Additionally, the sequential processing of batches isn't taking advantage of the multiple cores available on most modern systems.
            
            I recommend refactoring the data processing pipeline to use a streaming approach instead. This would involve reading and processing the data in chunks, which would significantly reduce memory usage. You could also implement parallel processing using the multiprocessing library to take advantage of multiple cores. I've seen similar implementations achieve a 3-4x speedup on comparable workloads.
            
            Another improvement would be to optimize the feature extraction functions. The current implementation recalculates certain features multiple times within the same processing pipeline. By caching intermediate results or restructuring the feature extraction process, you could eliminate this redundancy and further improve performance.
            """
        },
        {
            "title": "Meeting Notes Example",
            "text": """
            Today's project update meeting covered several key points about the new product launch scheduled for Q3. 
            
            The marketing team reported that the initial focus groups have been completed with positive feedback on the user interface. However, there were concerns about the pricing strategy which will need to be revisited before the launch. Sarah suggested conducting an additional survey with a larger sample group to validate the findings.
            
            Development reported that they're on track with the backend implementation but have encountered some integration issues with the payment provider API. They estimate a 1-week delay to resolve these issues but believe they can make up the time in the testing phase.
            
            Operations confirmed that the server infrastructure has been upgraded to handle the anticipated increase in traffic, and the new cloud deployment pipeline is working efficiently. They've also completed the disaster recovery plan as requested last month.
            
            The meeting concluded with agreement on the following action items: (1) Marketing to revise pricing strategy by June 15, (2) Development to provide daily updates on API integration progress, (3) All teams to review the updated launch timeline by end of week.
            """
        },
        {
            "title": "Product Description Example",
            "text": """
            The XDR-5000 is our most advanced security monitoring solution designed specifically for enterprise environments with complex infrastructure needs. This all-in-one platform integrates endpoint detection and response, network traffic analysis, and cloud security monitoring into a unified management console.
            
            Key features include real-time threat detection powered by our proprietary machine learning engine, which analyzes millions of security events per second to identify potential attacks before they cause damage. The system's automated response capabilities can isolate compromised endpoints, block malicious network connections, and revoke compromised credentials without manual intervention.
            
            The platform offers extensive customization options through its modular architecture, allowing security teams to enable only the components relevant to their environment. Its open API framework facilitates integration with existing security tools and SIEM solutions, ensuring a seamless fit into established security operations.
            
            Deployment can be on-premises, cloud-based, or hybrid, with dedicated support from our security engineers during implementation. The subscription includes 24/7 threat hunting services, quarterly security posture assessments, and continuous updates to threat intelligence feeds.
            """
        }
    ]
    
    # Initialize summarizer
    summarizer = TextSummarizer(model_name=args.model, use_gpu=args.use_gpu)
    
    # Test each example text
    for test_case in test_texts:
        title = test_case["title"]
        text = test_case["text"]
        
        print(f"\n\n{'='*80}")
        print(f"EXAMPLE: {title}")
        print(f"{'='*80}")
        
        print(f"\nOriginal text ({len(text)} chars):\n")
        print(text)
        
        # Test rule-based summarization
        rule_summary = summarizer.summarize(text, args.target_length, method="rule")
        print(f"\n--- RULE-BASED SUMMARY ({len(rule_summary)} chars) ---\n")
        print(rule_summary)
        
        # Test model-based summarization if available
        if summarizer._summarize_func is not None:
            try:
                model_summary = summarizer.summarize(text, args.target_length, method="model")
                print(f"\n--- MODEL-BASED SUMMARY ({len(model_summary)} chars) ---\n")
                print(model_summary)
            except Exception as e:
                print(f"\nError in model-based summarization: {e}")
    
    # Test conversation sparsification
    conversation = [
        {"role": "user", "content": "Can you help me optimize my data processing pipeline?"},
        {"role": "assistant", "content": test_texts[0]["text"]},
        {"role": "user", "content": "That's helpful. Could you provide some code examples?"},
        {"role": "assistant", "content": "Sure, here's an example of how you could implement the streaming approach with multiprocessing:\n\n```python\nimport pandas as pd\nfrom multiprocessing import Pool\n\ndef process_chunk(chunk):\n    # Process each chunk of data\n    result = chunk.apply(complex_calculation, axis=1)\n    return result\n\n# Read data in chunks\nchunks = pd.read_csv('large_dataset.csv', chunksize=10000)\n\n# Process chunks in parallel\nwith Pool(processes=4) as pool:\n    results = pool.map(process_chunk, chunks)\n\n# Combine results\nfinal_result = pd.concat(results)\n```\n\nThis approach has several advantages:\n1. Memory usage is limited to the chunk size\n2. Processing happens in parallel across multiple cores\n3. Each chunk is processed independently, allowing for better error handling"}
    ]
    
    print("\n\n" + "="*80)
    print("CONVERSATION SPARSIFICATION EXAMPLE")
    print("="*80)
    
    print("\n--- ORIGINAL CONVERSATION ---")
    total_length = 0
    for msg in conversation:
        print(f"\n{msg['role'].upper()}: ({len(msg['content'])} chars)")
        preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
        print(preview)
        total_length += len(msg['content'])
    print(f"\nTotal conversation length: {total_length} chars")
    
    # Sparsify conversation using model-based approach
    if summarizer._summarize_func is not None:
        try:
            sparsified = summarizer.sparsify_conversation(
                conversation.copy(), max_total_length=500)
            
            print("\n--- MODEL-BASED SPARSIFIED CONVERSATION ---")
            total_length = 0
            for msg in sparsified:
                print(f"\n{msg['role'].upper()}: ({len(msg['content'])} chars)")
                preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(preview)
                total_length += len(msg['content'])
            print(f"\nTotal sparsified length: {total_length} chars")
        except Exception as e:
            print(f"\nError in conversation sparsification: {e}")


if __name__ == "__main__":
    main() 