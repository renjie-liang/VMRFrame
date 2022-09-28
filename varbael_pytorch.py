SeqPAN(
  (text_encoder): Embedding(
    (word_emb): WordEmbedding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (char_emb): CharacterEmbedding(
      (char_emb): Embedding(36, 100, padding_idx=0)
      (char_convs): ModuleList(
        (0): Sequential(
          (0): Conv2d(100, 10, kernel_size=(1, 1), stride=(1, 1))
          (1): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(100, 20, kernel_size=(1, 2), stride=(1, 1))
          (1): ReLU()
        )
        (2): Sequential(
          (0): Conv2d(100, 30, kernel_size=(1, 3), stride=(1, 1))
          (1): ReLU()
        )
        (3): Sequential(
          (0): Conv2d(100, 40, kernel_size=(1, 4), stride=(1, 1))
          (1): ReLU()
        )
      )
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (query_conv1d): Conv1D(
      (conv1d): Conv1d(400, 128, kernel_size=(1,), stride=(1,))
    )
    (q_layer_norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
  )
  (video_affine): VisualProjection(
    (drop): Dropout(p=0.1, inplace=False)
    (video_conv1d): Conv1D(
      (conv1d): Conv1d(1024, 128, kernel_size=(1,), stride=(1,))
    )
    (v_layer_norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
  )
  (feat_encoder): FeatureEncoder(
    (pos_embedding): PositionalEmbedding(
      (position_embeddings): Embedding(64, 128)
    )
    (conv_block): DepthwiseSeparableConvBlock(
      (depthwise_separable_conv): ModuleList(
        (0): Sequential(
          (0): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), groups=128, bias=False)
          (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (2): ReLU()
        )
        (1): Sequential(
          (0): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), groups=128, bias=False)
          (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (2): ReLU()
        )
        (2): Sequential(
          (0): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), groups=128, bias=False)
          (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (2): ReLU()
        )
        (3): Sequential(
          (0): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), groups=128, bias=False)
          (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (2): ReLU()
        )
      )
      (layer_norms): ModuleList(
        (0): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        (1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        (2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        (3): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
      )
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (dual_attention_block_1): DualAttentionBlock(
    (dropout): Dropout(p=0.1, inplace=False)
    (layer_norm_1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    (layer_norm_2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    (layer_norm_t): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    (dense_1): Conv1D(
      (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
    )
    (dense_2): Conv1D(
      (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
    )
    (dual_multihead_attention): DualMultiAttention(
      (dropout): Dropout(p=0.1, inplace=False)
      (softmax): Softmax(dim=-1)
      (query): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (f_key): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (f_value): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (t_key): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (t_value): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (s_dense): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (x_dense): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (s_gate): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (x_gate): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (guided_dense): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (bilinear_1): BiLinear(
        (dense_1): Conv1D(
          (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        )
        (dense_2): Conv1D(
          (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (bilinear_2): BiLinear(
        (dense_1): Conv1D(
          (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        )
        (dense_2): Conv1D(
          (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (layer_norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
      (layer_norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
      (out_layer): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
    )
  )
  (dual_attention_block_2): DualAttentionBlock(
    (dropout): Dropout(p=0.1, inplace=False)
    (layer_norm_1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    (layer_norm_2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    (layer_norm_t): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    (dense_1): Conv1D(
      (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
    )
    (dense_2): Conv1D(
      (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
    )
    (dual_multihead_attention): DualMultiAttention(
      (dropout): Dropout(p=0.1, inplace=False)
      (softmax): Softmax(dim=-1)
      (query): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (f_key): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (f_value): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (t_key): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (t_value): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (s_dense): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (x_dense): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (s_gate): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (x_gate): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (guided_dense): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (bilinear_1): BiLinear(
        (dense_1): Conv1D(
          (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        )
        (dense_2): Conv1D(
          (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (bilinear_2): BiLinear(
        (dense_1): Conv1D(
          (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        )
        (dense_2): Conv1D(
          (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        )
      )
      (layer_norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
      (layer_norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
      (out_layer): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
    )
  )
  (q2v_attn): CQAttention(
    (dropout): Dropout(p=0.1, inplace=False)
    (cqa_linear): Conv1D(
      (conv1d): Conv1d(512, 128, kernel_size=(1,), stride=(1,))
    )
  )
  (v2q_attn): CQAttention(
    (dropout): Dropout(p=0.1, inplace=False)
    (cqa_linear): Conv1D(
      (conv1d): Conv1d(512, 128, kernel_size=(1,), stride=(1,))
    )
  )
  (cq_cat): CQConcatenate(
    (weighted_pool): WeightedPool()
    (conv1d): Conv1D(
      (conv1d): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
    )
  )
  (match_conv1d): Conv1D(
    (conv1d): Conv1d(128, 4, kernel_size=(1,), stride=(1,))
  )
  (predictor): SeqPANPredictor(
    (feature_encoder): FeatureEncoderPredict(
      (pos_embedding): PositionalEmbedding(
        (position_embeddings): Embedding(64, 128)
      )
      (conv_block): DepthwiseSeparableConvBlock(
        (depthwise_separable_conv): ModuleList(
          (0): Sequential(
            (0): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), groups=128, bias=False)
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): ReLU()
          )
          (1): Sequential(
            (0): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), groups=128, bias=False)
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): ReLU()
          )
          (2): Sequential(
            (0): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), groups=128, bias=False)
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): ReLU()
          )
          (3): Sequential(
            (0): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), groups=128, bias=False)
            (1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (2): ReLU()
          )
        )
        (layer_norms): ModuleList(
          (0): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (3): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        )
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (layer_norm_1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (layer_norm_2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
      (top_self_attention): TopSelfAttention2(
        (selfattn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
      )
      (dense): Conv1D(
        (conv1d): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
    )
    (start_layer_norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    (end_layer_norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
    (start_hidden): Conv1D(
      (conv1d): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
    )
    (end_hidden): Conv1D(
      (conv1d): Conv1d(256, 128, kernel_size=(1,), stride=(1,))
    )
    (start_dense): Conv1D(
      (conv1d): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
    )
    (end_dense): Conv1D(
      (conv1d): Conv1d(128, 1, kernel_size=(1,), stride=(1,))
    )
  )
)
