def exec(self):
    ''' Training End-to-end ASR system '''
    n_epochs = 0

    while self.step < self.max_step:
        for data in self.tr_set:
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            # Forward model
            ctc_output, encode_len, att_output, att_align, dec_state = \
                self.model(feat, feat_len, max(txt_len), tf_rate=tf_rate,
                           teacher=txt, get_dec_state=self.emb_reg)

            # Backprop
            grad_norm = self.backward(total_loss)
            self.step += 1

            # Logger
            if (self.step == 1) or (self.step % self.PROGRESS_STEP == 0):
                self.log()
            # Validation
            if (self.step == 1) or (self.step % self.valid_step == 0):
                self.validate()

            if self.step > self.max_step:
                break
        n_epochs += 1
