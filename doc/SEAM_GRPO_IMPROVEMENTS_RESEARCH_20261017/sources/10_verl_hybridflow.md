# HybridFlow: A Flexible and Efficient RLHF Framework (VeRL)

- Authors: Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, Chuan Wu (ByteDance Seed & HKU)
- Venue/year: EuroSys 2025 / `arXiv:2409.19256`, 2025.
- URL: https://arxiv.org/abs/2409.19256
- Repository: https://github.com/volcengine/verl
- Accessed: 2026-10-17.
- Type: peer-reviewed systems paper and production open-source framework.
- Credibility: very high. Recency: recent (2025–2026). Bias: low.

Verified excerpt:
> "We propose HybridFlow, which combines single-controller and multi-controller paradigms in a hybrid manner to enable flexible representation and efficient execution of the RLHF dataflow."

Relevance to SEAM GRPO:
Establishes the production blueprint for colocated actor + vLLM rollout engines: in-memory GPU weight transfer between optimizer steps and explicit memory lifecycle management (offloading/sleeping rollout buffers during the training backward pass).
