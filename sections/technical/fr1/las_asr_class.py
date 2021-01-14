class ASR(nn.Module):
    ''' ASR model, including Encoder/Decoder(s)'''

    def __init__(self, input_size, vocab_size, init_adadelta, ctc_weight, encoder, attention, decoder, emb_drop=0.0):
        super(ASR, self).__init__()

        # Modules
        self.encoder = Encoder(input_size, **encoder)
        self.dec_dim = decoder['dim']
        self.pre_embed = nn.Embedding(vocab_size, self.dec_dim)
        self.embed_drop = nn.Dropout(emb_drop)
        self.decoder = Decoder(
            self.encoder.out_dim+self.dec_dim, vocab_size, **decoder)
        query_dim = self.dec_dim*self.decoder.layer
        self.attention = Attention(
            self.encoder.out_dim, query_dim, **attention)
