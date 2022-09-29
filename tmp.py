
        # create mask for both visual and textual features
        v_mask = tf.sequence_mask(lengths=self.video_seq_len, maxlen=tf.reduce_max(self.video_seq_len), dtype=tf.int32) #(16, 64)
        q_mask = tf.cast(tf.cast(self.word_ids, dtype=tf.bool), dtype=tf.int32) #(16, 11)


        # generate query features
        #(16, 11, 300)
        word_emb = word_embs(self.word_ids, dim=self.configs.word_dim, drop_rate=self.drop_rate, finetune=False,
                             reuse=False, vectors=word_vectors)
        #(16, 11, 100)
        char_emb = char_embs(self.char_ids, char_size=self.configs.num_chars, dim=self.configs.char_dim, reuse=False,
                             kernels=[1, 2, 3, 4], filters=[10, 20, 30, 40], drop_rate=self.drop_rate, padding='VALID',
                             activation=tf.nn.relu)
        # (16, 11, 400)
        word_emb = tf.concat([word_emb, char_emb], axis=-1)


        
        qfeats = conv1d(word_emb, dim=self.configs.dim, use_bias=True, reuse=False, name='query_conv1d')
        qfeats = layer_norm(qfeats, reuse=False, name='q_layer_norm')
        # generate video features
        vfeats = tf.nn.dropout(self.video_inputs, rate=self.drop_rate)
        vfeats = conv1d(vfeats, dim=self.configs.dim, use_bias=True, reuse=False, name='video_conv1d')
        vfeats = layer_norm(vfeats, reuse=False, name='v_layer_norm')


        # add positional embedding
        vfeats = add_pos_embs(vfeats, max_pos_len=self.configs.max_pos_len, reuse=False, name='pos_emb')
        qfeats = add_pos_embs(qfeats, max_pos_len=self.configs.max_pos_len, reuse=True, name='pos_emb')
        # convolutional block
        vfeats = conv_block(vfeats, kernel_size=7, dim=self.configs.dim, num_layers=4, drop_rate=self.drop_rate,
                            activation=tf.nn.relu, reuse=False, name='conv_block')
        qfeats = conv_block(qfeats, kernel_size=7, dim=self.configs.dim, num_layers=4, drop_rate=self.drop_rate,
                            activation=tf.nn.relu, reuse=True, name='conv_block')
        # attention block
        for li in range(self.configs.attn_layer):
            vfeats_ = dual_attn_block(vfeats, qfeats, dim=self.configs.dim, num_heads=self.configs.num_heads,
                                      from_mask=v_mask, to_mask=q_mask, use_bias=True, drop_rate=self.drop_rate,
                                      activation=None, reuse=False, name='d_attn_%d' % li)
            qfeats_ = dual_attn_block(qfeats, vfeats, dim=self.configs.dim, num_heads=self.configs.num_heads,
                                      from_mask=q_mask, to_mask=v_mask, use_bias=True, drop_rate=self.drop_rate,
                                      activation=None, reuse=True, name='d_attn_%d' % li)
            vfeats = vfeats_
            qfeats = qfeats_
        # fuse features
        q2v_feats, _ = cq_attention(vfeats, qfeats, mask1=v_mask, mask2=q_mask, drop_rate=self.drop_rate,
                                    reuse=False, name='q2v_attn')
        v2q_feats, _ = cq_attention(qfeats, vfeats, mask1=q_mask, mask2=v_mask, drop_rate=self.drop_rate,
                                    reuse=False, name='v2q_attn')
        fuse_feats = cq_concat(q2v_feats, v2q_feats, pool_mask=q_mask, reuse=False, name='cq_cat')


        
        # compute matching loss and matching score
        # label_embs = tf.get_variable(name='label_emb', shape=[4, self.configs.dim], dtype=tf.float32,
        #                              trainable=True, initializer=tf.orthogonal_initializer())
        self.match_loss, match_scores = matching_loss(fuse_feats, self.match_labels, label_size=4, mask=v_mask,
                                                      gumbel=not self.configs.no_gumbel, tau=self.configs.tau,
                                                      reuse=False)

        label_embs = tf.get_variable(name='label_emb', shape=[4, self.configs.dim], dtype=tf.float32,
                                     trainable=True, initializer=tf.orthogonal_initializer())
        ortho_constraint = tf.multiply(tf.matmul(label_embs, label_embs, transpose_b=True),
                                       1.0 - tf.eye(4, dtype=tf.float32))
        ortho_constraint = tf.norm(ortho_constraint, ord=2)  # compute l2 norm as loss
        self.match_loss += ortho_constraint


        
        soft_label_embs = tf.matmul(match_scores, tf.tile(tf.expand_dims(label_embs, axis=0),
                                                          multiples=[tf.shape(match_scores)[0], 1, 1]))
        outputs = (fuse_feats + soft_label_embs) * tf.cast(tf.expand_dims(v_mask, axis=-1), dtype=tf.float32)
        # compute start and end logits
        self.start_logits, self.end_logits = conditioned_predictor(outputs, dim=self.configs.dim, reuse=False, mask=v_mask,
                                                         num_heads=self.configs.num_heads, drop_rate=self.drop_rate,
                                                         attn_drop=self.drop_rate, max_pos_len=self.configs.max_pos_len,
                                                         activation=tf.nn.relu, name="predictor")
        self.tmp1, self.tmp2 = self.start_logits, self.end_logits
        # compute localization loss
        self.loc_loss = localizing_loss(self.start_logits, self.end_logits, self.y1, self.y2, v_mask)
        # compute predicted indexes
        self.start_index, self.end_index = ans_predictor(self.start_logits, self.end_logits, v_mask)
        # total loss
        self.loss = self.loc_loss + self.configs.match_lambda * self.match_loss
        # create optimizer
        self.train_op = create_optimizer(self.loss, self.lr, clip_norm=self.configs.clip_norm)