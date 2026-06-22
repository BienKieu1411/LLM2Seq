"""
LLM2Seq: Converting LLM2Vec Encoders into Lightweight Encoder-Decoder Generators.

Core idea:
    Input x
      -> LLM2Vec Encoder: H_enc
      -> Adaptor: H_dec_memory
      -> Lightweight Decoder: P(y_t | y_<t, x)
      -> Optional MTP Heads / MTP Modules
      -> Output y
"""
