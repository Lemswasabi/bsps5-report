def load_data(self):
    self.tr_set, self.dv_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
        load_dataset(**self.config['data'])
