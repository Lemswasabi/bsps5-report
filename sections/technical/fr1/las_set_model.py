def set_model(self):
    ''' Setup ASR model and optimizer '''
    #model
    init_adadelta = self.config['hparas']['optimizer'] == 'Adadelta'
    self.model = ASR(self.feat_dim, self.vocab_size, init_adadelta, **
                     self.config['model']).to(self.device)

    # Losses
    self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
