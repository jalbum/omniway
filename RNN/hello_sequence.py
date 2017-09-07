"""Sequence-to-sequence model with an attention mechanism."""
# see https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html
# compare https://github.com/tflearn/tflearn/blob/master/examples/nlp/seq2seq_example.py
# https://github.com/Kyung-Min/Seq2Seq-TensorFlow
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

vocab_size=256 # We are lazy, so we avoid fency mapping and just use one *class* per character/byte
target_vocab_size=vocab_size
learning_rate=0.1
buckets=[(10, 10)] # our input and response words can be up to 10 characters long
PAD=[0] # fill words shorter than 10 characters with 'padding' zeroes
batch_size=10 # for parallel training (later)

# ord는 아스키코드로 변환함, PAD 0을 5개 채움, 그리고 batch사이즈 만큼 반복
# [[104, 101, 108, 108, 111, 0, 0],  .....10개.....  ,[104, 101, 108, 108, 111, 0, 0]]
# map은 파이선2에서 list로 돌림 파이선3에선 iterator로 돌리므로 파이선3에선 list함수로 변환
input_data    = [list(map(ord, "hello")) + PAD * 5 ] * batch_size
target_data   = [list(map(ord, "world")) + PAD * 5 ] * batch_size
print(input_data)

# The number of actual valid (loss counted) number of characters
#[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0] 을 배치사이즈만큼
target_weights= [[1.0]*6 + [0.0]*4] * batch_size # mask padding

EOS='\n' # end of sequence symbol todo use how?
GO=1		 # start symbol 0x01 todo use how?


class BabySeq2Seq(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, batch_size):
        self.buckets = buckets  #[(10,10)] 버켓은 1가지
        self.batch_size = batch_size # = 10
        self.source_vocab_size = source_vocab_size #= vacab_size =256
        self.target_vocab_size = target_vocab_size # = vacab_size =256

        #cell = single_cell = tf.nn.rnn_cell.GRUCell(size)
        cell = single_cell = tf.contrib.rnn.GRUCell(size)  # size=10
        if num_layers > 1: # 현재 num_layers는 1로 들어 옴
            # cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        # The seq2seq function
        # encoder_inputs: A list of ASCII codes in the input sentence.
        # decoder_inputs: A list of ASCII codes in the target sentence.
        # cell: RNN cell to use for seq2seq.
        # num_encoder_symbols, num_decoder_symbols: The number of symbols in the input sentence and the target sentence.
        # embedding_size: Size to embed each ASCII code.
        # feed_previous: Inference (true for learning / false for Inference)
        # 임베딩에 대한 개념은 아래
        # https://m.blog.naver.com/PostView.nhn?blogId=godkad&logNo=220990414180&proxyReferer=http%3A%2F%2Fwww.google.co.kr%2Furl%3Fsa%3Di%26rct%3Dj%26q%3D%26esrc%3Ds%26source%3Dimages%26cd%3D%26ved%3D0ahUKEwiI6PTcw_nVAhUCi7wKHfBcBtMQjhwIBQ%26url%3Dhttp%253A%252F%252Fm.blog.naver.com%252Fgodkad%252F220990414180%26psig%3DAFQjCNFZl7It2r8VnIN-qdSzS2AVwpn0tQ%26ust%3D1503995948124537
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            # tf.nn.seq2seq.embedding_attention_seq2seq 변경 => tf.contrib.legacy_seq2seq.embedding_attention_seq2seq
            return  tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell, # encoder_inputs=x decoder_inputs = y, cell : GRUCell
                num_encoder_symbols=source_vocab_size, #= vacab_size =256
                num_decoder_symbols=target_vocab_size, #= vacab_size =256
                embedding_size=size, # size=10
                feed_previous=do_decode) #do_decode=False

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one. OK
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]
        # tf.nn.seq2seq.model_with_buckets 변경 => tf.contrib.legacy_seq2seq.model_with_buckets
        self.outputs, self.losses =  tf.contrib.legacy_seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, buckets,
            lambda x, y: seq2seq_f(x, y, False))

        # Gradients update operation for training the model.
        params = tf.trainable_variables()
        self.updates = []
        for b in xrange(len(buckets)):
            self.updates.append(tf.train.AdamOptimizer(learning_rate).minimize(self.losses[b]))

        # self.saver = tf.train.Saver(tf.all_variables())
        # WARNING:tensorflow:From C:/Users/9562916/PycharmProjects/RNN/TensTest1.py:83: all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
        #         Instructions for updating:
        # Please use tf.global_variables instead.
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, test):
        bucket_id = 0  # todo: auto-select
        encoder_size, decoder_size = self.buckets[bucket_id]

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not test:
            output_feed = [self.updates[bucket_id], self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not test:
            return outputs[0], outputs[1]  # Gradient norm, loss
        else:
            return outputs[0], outputs[1:]  # loss, outputs.

def decode(bytes):
    return "".join(map(chr, bytes)).replace('\x00', '').replace('\n', '')


def test(sess):
    perplexity, outputs = model.step(sess, input_data, target_data, target_weights, test=True)
    words = np.argmax(outputs, axis=2)  # shape (10, 10, 256)
    word = decode(words[0])
    # terminal should be UTF-8
    # on MS Windows, run chcp 65001
    print("step %d, perplexity %f, output: hello %s?" % (step, perplexity, word))
    if word == "world":
        print(">>>>> success! hello " + word + "! <<<<<<<")
        return True
    return False

step = 0
test_step = 1
with tf.Session() as sess:
    model = BabySeq2Seq(vocab_size, target_vocab_size, buckets, size=10, num_layers=1, batch_size=batch_size)
    sess.run(tf.global_variables_initializer())
    finished = False
    while not finished:
        model.step(sess, input_data, target_data, target_weights, test=False)        # no outputs in training
        if step % test_step == 0:
            finished = test(sess)
        step = step + 1