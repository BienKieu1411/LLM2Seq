import argparse

import gradio as gr
from transformers import AutoTokenizer, EncoderDecoderModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model = EncoderDecoderModel.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    def summarize(text, max_new_tokens, num_beams):
        if not text.strip():
            return ""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    with gr.Blocks(title="Encoder-Decoder Summarization Demo") as demo:
        gr.Markdown("# Encoder-Decoder LLM Summarization")
        inp = gr.Textbox(lines=14, label="Input Document")
        with gr.Row():
            max_new_tokens = gr.Slider(32, 256, value=128, step=8, label="Summary Max Tokens")
            num_beams = gr.Slider(1, 8, value=4, step=1, label="Beam Size")
        out = gr.Textbox(lines=8, label="Summary")
        btn = gr.Button("Summarize")
        btn.click(summarize, inputs=[inp, max_new_tokens, num_beams], outputs=out)

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
