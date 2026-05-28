import random

def apply_prefix_lm(text: str, prefix_ratio: float = 0.5) -> tuple[str, str]:
    """
    A basic PrefixLM preparation that splits text into context and continuation.
    Not heavily used when finetuning on summarization directly, but can be
    used for continued pretraining.
    """
    words = text.split()
    if len(words) < 4:
        return text, ""
    
    split_idx = int(len(words) * prefix_ratio)
    prefix_text = " ".join(words[:split_idx])
    target_text = " ".join(words[split_idx:])
    
    return prefix_text, target_text

def apply_ul2_s_denoising(text: str, mask_token: str = "<extra_id_0>") -> tuple[str, str]:
    """
    S-Denoising (Sequential Denoising) from UL2.
    Very simple heuristic: drop out a chunk.
    """
    words = text.split()
    if len(words) < 5:
        return text, ""
        
    start_idx = random.randint(0, len(words) // 2)
    end_idx = start_idx + random.randint(2, len(words) // 3)
    
    dropped = words[start_idx:end_idx]
    
    source = words[:start_idx] + [mask_token] + words[end_idx:]
    target = [mask_token] + dropped
    
    return " ".join(source), " ".join(target)
