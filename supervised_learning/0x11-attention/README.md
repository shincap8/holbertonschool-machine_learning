# 0x11. Attention

### Description

This project is about implementing attention mechanism.

### General Objectives

* What is the attention mechanism?
* How to apply attention to RNNs
* What is a transformer?
* How to create an encoder-decoder transformer model
* What is GPT?
* What is BERT?
* What is self-supervised learning?
* How to use BERT for specific NLP tasks
* What is SQuAD? GLUE?

### Mandatory Tasks

| File | Description |
| ------ | ------ |
| [0-rnn_encoder.py](0-rnn_encoder.py) | RNNEncoder class that inherits from tensorflow.keras.layers.Layer to encode for machine translation. |
| [1-self_attention.py](1-self_attention.py) | SelfAttention class that inherits from tensorflow.keras.layers.Layer to calculate the attention for machine translation. |
| [2-rnn_decoder.py](2-rnn_decoder.py) | RNNDecoder class that inherits from tensorflow.keras.layers.Layer to decode for machine translation:. |
| [4-positional_encoding.py](4-positional_encoding.py) | Calculates the positional encoding for a transformer. |
| [5-sdp_attention.py](5-sdp_attention.py) | Calculates the scaled dot product attention. |
| [6-multihead_attention.py](6-multihead_attention.py) | MultiHeadAttention class that inherits from tensorflow.keras.layers.Layer to perform multi head attention. |
| [7-transformer_encoder_block.py](7-transformer_encoder_block.py) | EncoderBlock class that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer. |
| [8-transformer_decoder_block.py](8-transformer_decoder_block.py) | DecoderBlock class that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer. |
| [9-transformer_encoder.py](9-transformer_encoder.py) | Encoder class that inherits from tensorflow.keras.layers.Layer to create the encoder for a transformer. |
| [10-transformer_decoder.py](10-transformer_decoder.py) | Decoder class that inherits from tensorflow.keras.layers.Layer to create the decoder for a transformer. |
| [11-transformer.py](11-transformer.py) | Transformer class that inherits from tensorflow.keras.Model to create a transformer network. |
