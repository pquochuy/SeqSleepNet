class Config(object):
    def __init__(self):
        # input frequency dimension,
        self.ndim = 129
        # number of frames per epoch
        self.frame_seq_len = 29
        # number of epochs per sequence
        self.epoch_seq_len = 20
        # number of channels
        self.nchannel = 1
        self.nclass = 5

        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.001
        self.training_epoch = 10
        self.batch_size = 32
        #self.batch_size_per_class = 6

        self.dropout_keep_prob_rnn = 0.75

        self.frame_step = self.frame_seq_len
        self.epoch_step = self.epoch_seq_len
        self.nhidden1 = 64
        self.nlayer1 = 1
        self.attention_size1 = 64
        self.nhidden2 = 64
        self.nlayer2 = 1

        self.nfilter = 32
        self.nfft = 256
        self.samplerate = 100
        self.lowfreq = 0
        self.highfreq = 50

        self.evaluate_every = 100
        self.checkpoint_every = 100
