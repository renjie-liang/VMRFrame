'word_embs/unk:0' shape=(1, 300) dtype=float32_ref>
'char_embs/char_table:0' shape=(35, 50) dtype=float32_ref>
'char_embs/filter_0:0' shape=(1, 1, 50, 10) dtype=float32_ref>
'char_embs/bias_0:0' shape=(10,) dtype=float32_ref>
'char_embs/filter_1:0' shape=(1, 2, 50, 20) dtype=float32_ref>
'char_embs/bias_1:0' shape=(20,) dtype=float32_ref>
'char_embs/filter_2:0' shape=(1, 3, 50, 30) dtype=float32_ref>
'char_embs/bias_2:0' shape=(30,) dtype=float32_ref>
'char_embs/filter_3:0' shape=(1, 4, 50, 40) dtype=float32_ref>
'char_embs/bias_3:0' shape=(40,) dtype=float32_ref>
'query_conv1d/kernel:0' shape=(1, 400, 128) dtype=float32_ref>
'query_conv1d/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'q_layer_norm/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'q_layer_norm/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'video_conv1d/kernel:0' shape=(1, 1024, 128) dtype=float32_ref>
'video_conv1d/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'v_layer_norm/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'v_layer_norm/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'pos_emb/position_embeddings:0' shape=(64, 128) dtype=float32_ref>
'conv_block/layer_norm_0/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'conv_block/layer_norm_0/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'conv_block/depthwise_conv_layers_0/depthwise_filter:0' shape=(7, 1, 128, 1) dtype=float32_ref>
'conv_block/depthwise_conv_layers_0/pointwise_filter:0' shape=(1, 1, 128, 128) dtype=float32_ref>
'conv_block/depthwise_conv_layers_0/bias:0' shape=(128,) dtype=float32_ref>
'conv_block/layer_norm_1/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'conv_block/layer_norm_1/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'conv_block/depthwise_conv_layers_1/depthwise_filter:0' shape=(7, 1, 128, 1) dtype=float32_ref>
'conv_block/depthwise_conv_layers_1/pointwise_filter:0' shape=(1, 1, 128, 128) dtype=float32_ref>
'conv_block/depthwise_conv_layers_1/bias:0' shape=(128,) dtype=float32_ref>
'conv_block/layer_norm_2/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'conv_block/layer_norm_2/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'conv_block/depthwise_conv_layers_2/depthwise_filter:0' shape=(7, 1, 128, 1) dtype=float32_ref>
'conv_block/depthwise_conv_layers_2/pointwise_filter:0' shape=(1, 1, 128, 128) dtype=float32_ref>
'conv_block/depthwise_conv_layers_2/bias:0' shape=(128,) dtype=float32_ref>
'conv_block/layer_norm_3/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'conv_block/layer_norm_3/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'conv_block/depthwise_conv_layers_3/depthwise_filter:0' shape=(7, 1, 128, 1) dtype=float32_ref>
'conv_block/depthwise_conv_layers_3/pointwise_filter:0' shape=(1, 1, 128, 128) dtype=float32_ref>
'conv_block/depthwise_conv_layers_3/bias:0' shape=(128,) dtype=float32_ref>
'd_attn_0/layer_norm_1/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'd_attn_0/layer_norm_1/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'd_attn_0/layer_norm_t/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'd_attn_0/layer_norm_t/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/query/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/query/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/f_key/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/f_key/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/f_value/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/f_value/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/t_key/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/t_key/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/t_value/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/t_value/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/s_dense/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/s_dense/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/x_dense/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/x_dense/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/s_gate/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/s_gate/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/x_gate/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/x_gate/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/guided_dense/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/guided_dense/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/bilinear_1/dense_1/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/bilinear_1/dense_2/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/bilinear_1/bias:0' shape=(128,) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/bilinear_2/dense_1/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/bilinear_2/dense_2/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dual_multihead_attention/bilinear_2/bias:0' shape=(128,) dtype=float32_ref>
'd_attn_0/dense_1/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dense_1/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_0/layer_norm_2/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'd_attn_0/layer_norm_2/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'd_attn_0/dense_2/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_0/dense_2/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/layer_norm_1/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'd_attn_1/layer_norm_1/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'd_attn_1/layer_norm_t/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'd_attn_1/layer_norm_t/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/query/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/query/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/f_key/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/f_key/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/f_value/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/f_value/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/t_key/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/t_key/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/t_value/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/t_value/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/s_dense/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/s_dense/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/x_dense/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/x_dense/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/s_gate/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/s_gate/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/x_gate/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/x_gate/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/guided_dense/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/guided_dense/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/bilinear_1/dense_1/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/bilinear_1/dense_2/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/bilinear_1/bias:0' shape=(128,) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/bilinear_2/dense_1/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/bilinear_2/dense_2/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dual_multihead_attention/bilinear_2/bias:0' shape=(128,) dtype=float32_ref>
'd_attn_1/dense_1/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dense_1/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'd_attn_1/layer_norm_2/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'd_attn_1/layer_norm_2/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'd_attn_1/dense_2/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'd_attn_1/dense_2/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'q2v_attn/efficient_trilinear/linear_kernel4arg0:0' shape=(128, 1) dtype=float32_ref>
'q2v_attn/efficient_trilinear/linear_kernel4arg1:0' shape=(128, 1) dtype=float32_ref>
'q2v_attn/efficient_trilinear/linear_kernel4mul:0' shape=(1, 1, 128) dtype=float32_ref>
'q2v_attn/dense/kernel:0' shape=(1, 512, 128) dtype=float32_ref>
'v2q_attn/efficient_trilinear/linear_kernel4arg0:0' shape=(128, 1) dtype=float32_ref>
'v2q_attn/efficient_trilinear/linear_kernel4arg1:0' shape=(128, 1) dtype=float32_ref>
'v2q_attn/efficient_trilinear/linear_kernel4mul:0' shape=(1, 1, 128) dtype=float32_ref>
'v2q_attn/dense/kernel:0' shape=(1, 512, 128) dtype=float32_ref>
'cq_cat/weighted_pooling/weight:0' shape=(128, 1) dtype=float32_ref>
'cq_cat/dense/kernel:0' shape=(1, 256, 128) dtype=float32_ref>
'cq_cat/dense/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'label_emb:0' shape=(4, 128) dtype=float32_ref>
'matching_loss/dense/kernel:0' shape=(1, 128, 4) dtype=float32_ref>
'matching_loss/dense/bias:0' shape=(1, 1, 4) dtype=float32_ref>


'predictor/feature_encoder/pos_emb/position_embeddings:0' shape=(64, 128) dtype=float32_ref>
'predictor/feature_encoder/conv_block/layer_norm_0/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/conv_block/layer_norm_0/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_0/depthwise_filter:0' shape=(7, 1, 128, 1) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_0/pointwise_filter:0' shape=(1, 1, 128, 128) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_0/bias:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/conv_block/layer_norm_1/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/conv_block/layer_norm_1/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_1/depthwise_filter:0' shape=(7, 1, 128, 1) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_1/pointwise_filter:0' shape=(1, 1, 128, 128) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_1/bias:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/conv_block/layer_norm_2/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/conv_block/layer_norm_2/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_2/depthwise_filter:0' shape=(7, 1, 128, 1) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_2/pointwise_filter:0' shape=(1, 1, 128, 128) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_2/bias:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/conv_block/layer_norm_3/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/conv_block/layer_norm_3/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_3/depthwise_filter:0' shape=(7, 1, 128, 1) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_3/pointwise_filter:0' shape=(1, 1, 128, 128) dtype=float32_ref>
'predictor/feature_encoder/conv_block/depthwise_conv_layers_3/bias:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/multihead_attention_block/layer_norm_1/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/multihead_attention_block/layer_norm_1/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/multihead_attention_block/top_self_attention/query/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'predictor/feature_encoder/multihead_attention_block/top_self_attention/query/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'predictor/feature_encoder/multihead_attention_block/top_self_attention/key/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'predictor/feature_encoder/multihead_attention_block/top_self_attention/key/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'predictor/feature_encoder/multihead_attention_block/top_self_attention/value/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'predictor/feature_encoder/multihead_attention_block/top_self_attention/value/bias:0' shape=(1, 1, 128) dtype=float32_ref>
'predictor/feature_encoder/multihead_attention_block/layer_norm_2/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'predictor/feature_encoder/multihead_attention_block/layer_norm_2/layer_norm_bias:0' shape=(128,) dtype=float32_ref>


'predictor/feature_encoder/multihead_attention_block/dense/kernel:0' shape=(1, 128, 128) dtype=float32_ref>
'predictor/feature_encoder/multihead_attention_block/dense/bias:0' shape=(1, 1, 128) dtype=float32_ref>




'predictor/start_layer_norm/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'predictor/start_layer_norm/layer_norm_bias:0' shape=(128,) dtype=float32_ref>
'predictor/end_layer_norm/layer_norm_scale:0' shape=(128,) dtype=float32_ref>
'predictor/end_layer_norm/layer_norm_bias:0' shape=(128,) dtype=float32_ref>

'predictor/start_hidden/kernel:0' shape=(1, 256, 128) dtype=float32_ref>
'predictor/start_hidden/bias:0' shape=(1, 1, 128) dtype=float32_ref>

'predictor/end_hidden/kernel:0' shape=(1, 256, 128) dtype=float32_ref>
'predictor/end_hidden/bias:0' shape=(1, 1, 128) dtype=float32_ref>

'predictor/start_dense/kernel:0' shape=(1, 128, 1) dtype=float32_ref>
'predictor/start_dense/bias:0' shape=(1, 1, 1) dtype=float32_ref>

'predictor/end_dense/kernel:0' shape=(1, 128, 1) dtype=float32_ref>
'predictor/end_dense/bias:0' shape=(1, 1, 1) dtype=float32_ref>