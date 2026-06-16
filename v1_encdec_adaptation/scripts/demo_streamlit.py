import argparse
import time
import torch
import streamlit as st
from transformers import AutoTokenizer, EncoderDecoderModel

# Setup page config
st.set_page_config(page_title="VietNews Summarization", page_icon="📝", layout="wide")

@st.cache_resource
def load_model(model_dir: str):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = EncoderDecoderModel.from_pretrained(model_dir).to(device)
    return model, tokenizer, device

def summarize(text: str, model_dir: str, max_new_tokens: int, num_beams: int, no_repeat_ngram_size: int):
    if not text.strip():
        return "", 0.0, 0, 0
    
    model, tokenizer, device = load_model(model_dir)
    
    start_time = time.time()
    
    # Prefix
    input_text = "vietnews: " + text.strip()
    
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048
    ).to(device)
    
    out = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens, 
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=False,
        early_stopping=True
    )
    
    summary = tokenizer.decode(out[0], skip_special_tokens=True)
    inference_time = time.time() - start_time
    
    input_tokens = inputs.input_ids.shape[1]
    output_tokens = out.shape[1]
    
    return summary, inference_time, input_tokens, output_tokens

def main():
    st.title("📝 Encoder-Decoder LLM cho Tóm tắt Báo chí Tiếng Việt")
    st.markdown("""
    Ứng dụng Demo sử dụng mô hình LLM đã được chuyển đổi từ **Decoder-Only** sang **Encoder-Decoder**, 
    tối ưu cho bài toán tóm tắt văn bản dài.
    """)
    
    # Sidebar configs
    with st.sidebar:
        st.header("Cấu hình Mô hình")
        model_dir = st.text_input("Đường dẫn mô hình", value="outputs/smoke-ed-cnn")
        
        st.header("Tham số Giải mã (Beam Search)")
        max_new_tokens = st.slider("Max Tokens Output", 64, 512, 256, 16)
        num_beams = st.slider("Beam Size", 1, 8, 5, 1)
        no_repeat_ngram = st.slider("No Repeat N-Gram", 0, 5, 3, 1)
        
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Văn bản nguồn (Bài báo)")
        source_text = st.text_area("Nhập hoặc dán văn bản tại đây...", height=400)
        
        if st.button("Tóm tắt", type="primary", use_container_width=True):
            if source_text:
                with st.spinner("Đang chạy mô hình..."):
                    summary, inf_time, in_toks, out_toks = summarize(
                        source_text, model_dir, max_new_tokens, num_beams, no_repeat_ngram
                    )
                    st.session_state['summary'] = summary
                    st.session_state['stats'] = {
                        'time': inf_time,
                        'in_toks': in_toks,
                        'out_toks': out_toks
                    }
            else:
                st.warning("Vui lòng nhập văn bản.")

    with col2:
        st.subheader("Bản tóm tắt")
        if 'summary' in st.session_state:
            st.info(st.session_state['summary'])
            
            st.divider()
            stats = st.session_state['stats']
            st.markdown(f"⏱️ **Thời gian suy diễn:** `{stats['time']:.2f} s`")
            st.markdown(f"📊 **Token đầu vào:** `{stats['in_toks']}` | **Token đầu ra:** `{stats['out_toks']}`")
            st.markdown(f"🚀 **Tốc độ sinh:** `{stats['out_toks']/stats['time']:.2f} tokens/s`")

if __name__ == "__main__":
    main()
