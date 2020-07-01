import tensorflow as tf
from nn_basic_layers import *
from filterbank_shape import FilterbankShape

class Multitask_ARNN_Sleep(object):
    """
    Multitask attentional recurrent neural network for sleep staging
    """

    def __init__(self, config):
        # Placeholders for input, output and dropout
        self.config = config
        self.input_x = tf.placeholder(tf.float32, [None, self.config.frame_step, self.config.ndim, self.config.nchannel], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.epoch_step, self.config.nclass], name="input_y")
        self.dropout_keep_prob_rnn = tf.placeholder(tf.float32, name="dropout_keep_prob_rnn")

        self.frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN

        filtershape = FilterbankShape()
        #triangular filterbank
        self.Wbl = tf.constant(filtershape.lin_tri_filter_shape(nfilt=self.config.nfilter,
                                                                nfft=self.config.nfft,
                                                                samplerate=self.config.samplerate,
                                                                lowfreq=self.config.lowfreq,
                                                                highfreq=self.config.highfreq),
                               dtype=tf.float32,
                               name="W-filter-shape-eeg")

        with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-eeg"):
            # Temporarily crush the feature_mat's dimensions
            Xeeg = tf.reshape(tf.squeeze(self.input_x[:,:,:,0]), [-1, self.config.ndim])
            # first filter bank layer
            self.Weeg = tf.Variable(tf.random_normal([self.config.ndim, self.config.nfilter],dtype=tf.float32))
            # non-negative constraints
            self.Weeg = tf.sigmoid(self.Weeg)
            # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
            self.Wfb = tf.multiply(self.Weeg,self.Wbl)
            HWeeg = tf.matmul(Xeeg, self.Wfb) # filtering
            HWeeg = tf.reshape(HWeeg, [-1, self.config.frame_step, self.config.nfilter])

        if(self.config.nchannel > 1):
            with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-eog"):
                # Temporarily crush the feature_mat's dimensions
                Xeog = tf.reshape(tf.squeeze(self.input_x[:,:,:,1]), [-1, self.config.ndim])
                # first filter bank layer
                self.Weog = tf.Variable(tf.random_normal([self.config.ndim, self.config.nfilter],dtype=tf.float32))
                # non-negative constraints
                self.Weog = tf.sigmoid(self.Weog)
                # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
                self.Wfb = tf.multiply(self.Weog,self.Wbl)
                HWeog = tf.matmul(Xeog, self.Wfb) # filtering
                HWeog = tf.reshape(HWeog, [-1, self.config.frame_step, self.config.nfilter])

        if(self.config.nchannel > 2):
            with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-emg"):
                # Temporarily crush the feature_mat's dimensions
                Xemg = tf.reshape(tf.squeeze(self.input_x[:,:,:,2]), [-1, self.config.ndim])
                # first filter bank layer
                self.Wemg = tf.Variable(tf.random_normal([self.config.ndim, self.config.nfilter],dtype=tf.float32))
                # non-negative constraints
                self.Wemg = tf.sigmoid(self.Wemg)
                # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
                self.Wfb = tf.multiply(self.Wemg,self.Wbl)
                HWemg = tf.matmul(Xemg, self.Wfb) # filtering
                HWemg = tf.reshape(HWemg, [-1, self.config.frame_step, self.config.nfilter])

        if(self.config.nchannel > 2):
            X = tf.concat([HWeeg, HWeog, HWemg], axis = 2)
        elif(self.config.nchannel > 1):
            X = tf.concat([HWeeg, HWeog], axis = 2)
        else:
            X = HWeeg

        # bidirectional frame-level recurrent layer
        with tf.device('/gpu:0'), tf.variable_scope("frame_rnn_layer") as scope:
            fw_cell1, bw_cell1 = bidirectional_recurrent_layer(self.config.nhidden1,
                                                                  self.config.nlayer1,
                                                                  input_keep_prob=self.dropout_keep_prob_rnn,
                                                                  output_keep_prob=self.dropout_keep_prob_rnn)
            rnn_out1, rnn_state1 = bidirectional_recurrent_layer_output(fw_cell1,
                                                                        bw_cell1,
                                                                        X,
                                                                        self.frame_seq_len,
                                                                        scope=scope)
            print(rnn_out1.get_shape())
            # output shape (batchsize*epoch_step, frame_step, nhidden1*2)


        with tf.device('/gpu:0'), tf.variable_scope("frame_attention_layer"):
            self.attention_out1 = attention(rnn_out1, self.config.attention_size1)
            print(self.attention_out1.get_shape())
            # attention_output1 of shape (batchsize, nhidden1*2)

        self.scores = []
        self.predictions = []
        with tf.device('/gpu:0'), tf.variable_scope("output_layer"):
            for i in range(self.config.epoch_step):
                score_i = fc(self.attention_out1,
                                self.config.nhidden1 * 2,
                                self.config.nclass,
                                name="output-%s" % i,
                                relu=False)
                pred_i = tf.argmax(score_i, 1, name="pred-%s" % i)
                self.scores.append(score_i)
                self.predictions.append(pred_i)

        # calculate cross-entropy output loss
        self.output_loss = 0
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            for i in range(self.config.epoch_step):
                output_loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(self.input_y[:,i,:]), logits=self.scores[i])
                output_loss_i = tf.reduce_sum(output_loss_i, axis=[0])
                self.output_loss += output_loss_i

            # add on regularization
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            except_vars_eeg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-eeg')
            except_vars_eog = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-eog')
            except_vars_emg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='filterbank-layer-emg')
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if v not in except_vars_eeg and v not in except_vars_eog and v not in except_vars_emg])
            self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss

        self.accuracy = []
        # Accuracy
        with tf.device('/gpu:0'), tf.name_scope("accuracy"):
            for i in range(self.config.epoch_step):
                correct_prediction_i = tf.equal(self.predictions[i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                accuracy_i = tf.reduce_mean(tf.cast(correct_prediction_i, "float"), name="accuracy-%s" % i)
                self.accuracy.append(accuracy_i)

