import argparse
import yaml
import os

from src.adaptation import AdaptationConfig, save_adapted_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--output_dir", required=True, help="Output directory")
    return p.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    cfg = AdaptationConfig(
        encoder_model=config_data["encoder_model"],
        decoder_model=config_data.get("decoder_model", None),
        output_dir=args.output_dir,
        max_length=config_data.get("max_length", 1024),
        tied_embeddings=config_data.get("tied_embeddings", False),
        cross_attn_init=config_data.get("cross_attn_init", "from_self_attention"),
        warmup_steps=config_data.get("warmup_steps", 0)
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_adapted_model(cfg)
    print(f"Saved adapted encoder-decoder model to: {args.output_dir}")

if __name__ == "__main__":
    main()
