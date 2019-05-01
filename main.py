import model
import tensorflow as tf
from data import load_pieces, build_vocab, \
                 tokenize, get_finetuning_batch, get_pretraining_batch
import numpy as np
import sys
from tqdm import tqdm
import hyper_params as hp
from midi_handler import noteStateMatrixToMidi

np.set_printoptions(threshold=sys.maxsize)


def train(model,
          pieces,
          token2idx,
          epochs,
          save_name,
          load_name=None,
          mask_prob=0.15,
          finetuning=False):
    sess = tf.Session()
    if load_name is not None:
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(load_name + '.meta')
        saver.restore(sess, load_name)
    else:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
    pbar = tqdm(range(0, epochs))
    for i in pbar:
        if finetuning:
            x, y, mask = get_finetuning_batch(pieces,
                                              token2idx,
                                              hp.BATCH_SIZE)
        else:
            x, y, mask = get_pretraining_batch(pieces,
                                               token2idx,
                                               hp.BATCH_SIZE,
                                               mask_prob=mask_prob)
        l, _ = sess.run([model.loss, model.optimize],
                        feed_dict={model.inputs: x,
                                   model.labels: y,
                                   model.mask: mask,
                                   model.dropout: hp.KEEP_PROB})
        pbar.set_description("epoch {}, loss={}".format(i, l))
        if i % 100 == 0:
            print("epoch {}, loss={}".format(i, l))
        if i % 500 == 0:
            print("Saving at epoch {}, loss={}".format(i, l))
            saver.save(sess,
                       save_name + str(l),
                       global_step=i)
        if i % 1000 == 0:
            total_correct = 0
            total_symbols = 0
            for j in range(100):
                if finetuning:
                    x, y, mask = get_finetuning_batch(pieces,
                                                      token2idx,
                                                      hp.BATCH_SIZE,
                                                      testing=True)
                else:
                    x, y, mask = get_pretraining_batch(pieces,
                                                       token2idx,
                                                       hp.BATCH_SIZE,
                                                       mask_prob=mask_prob,
                                                       testing=True)
                prediction = sess.run(model.logits,
                                      feed_dict={model.inputs: x,
                                                 model.dropout: 1.0})
                activation = np.argmax(prediction, axis=2)
                # print("act: ", activation)
                # print("lab: ", y)
                total_correct += np.sum(np.multiply(y == activation, mask))
                total_symbols += np.sum(mask)
            print(total_correct / total_symbols)
    final_loss = sess.run([model.loss],
                          feed_dict={model.inputs: x,
                                     model.labels: y})
    saver.save(sess, save_name + str(final_loss[0]))


def test(model, pieces, save_name, finetuning=True):
    sess = tf.Session()
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(save_name + '.meta')
    saver.restore(sess, save_name)
    total_correct = 0
    total_symbols = 0
    for j in range(100):
        if finetuning:
            x, y, mask = get_finetuning_batch(pieces,
                                              token2idx,
                                              hp.BATCH_SIZE,
                                              testing=False)
        else:
            x, y, mask = get_pretraining_batch(pieces,
                                               token2idx,
                                               hp.BATCH_SIZE,
                                               mask_prob=0.15,
                                               testing=False)
        # print(x[0])
        # print(y[0])
        # print(mask[0])
        prediction = sess.run(model.logits,
                              feed_dict={model.inputs: x,
                                         model.dropout: 1.0})
        activation = np.argmax(prediction, axis=2)
        # print(activation[0])
        # print("lab: ", y)
        total_correct += np.sum(np.multiply(y == activation, mask))
        total_symbols += np.sum(mask)
    print(total_correct / total_symbols)


def generate(model,
             pieces,
             save_name,
             token2idx,
             idx2token,
             batch_size=10,
             length=1000):
    sess = tf.Session()
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(save_name + '.meta')
    saver.restore(sess, save_name)
    x, y, mask = get_finetuning_batch(pieces,
                                      token2idx,
                                      1,
                                      testing=True)
    # (batch_size, max_len, pitch_sz)
    composition = np.zeros((x.shape[0],
                            int(x.shape[1] / 4),
                            hp.NOTE_LEN, 2))
    real_comp = np.zeros((x.shape[0],
                          int(x.shape[1] / 4),
                          hp.NOTE_LEN, 2))
    previous = np.zeros((hp.NOTE_LEN, 2))
    real_previous = np.zeros((hp.NOTE_LEN, 2))
    # (batch_size, max_len, vocab_size)
    prediction = sess.run(model.logits,
                          feed_dict={model.inputs: x,
                                     model.dropout: 1.0})
    # (batch_size, max_len)
    activation = np.argmax(prediction, axis=2)
    for i in range(4, int(x.shape[1] / 4)):
        for j in range(4):
            pitch = idx2token[activation[0][i*4 + j-1]] - 24
            if pitch < hp.NOTE_LEN:
                composition[0][i][pitch][0] = 1
                if previous[pitch][0] == 1:
                    composition[0][i][pitch][1] = 0
                else:
                    composition[0][i][pitch][1] = 1

            real_pitch = idx2token[y[0][i*4 + j]] - 24
            if real_pitch < hp.NOTE_LEN:
                real_comp[0][i][real_pitch][0] = 1
                if real_previous[real_pitch][0] == 1:
                    real_comp[0][i][real_pitch][1] = 0
                else:
                    real_comp[0][i][real_pitch][1] = 1
        previous = composition[0][i]
        real_previous = real_comp[0][i]
    print(composition.shape)
    for song_idx in range(composition.shape[0]):
        noteStateMatrixToMidi(composition[song_idx],
                              'output/sample_' + str(song_idx))
    for song_idx in range(real_comp.shape[0]):
        noteStateMatrixToMidi(real_comp[song_idx],
                              'output/real_sample_' + str(song_idx))


if __name__ == '__main__':
    inputs = tf.placeholder(tf.int32, shape=[None, hp.MAX_LEN])
    labels = tf.placeholder(tf.int32, shape=[None, hp.MAX_LEN])
    mask = tf.placeholder(tf.float32, shape=[None, hp.MAX_LEN])
    dropout = tf.placeholder(tf.float32, shape=())

    pieces, seqlens = load_pieces("data/roll/jsb8.pkl")
    token2idx, idx2token = build_vocab(pieces)
    pieces = tokenize(pieces, token2idx, idx2token)
    m = model.Model(inputs=inputs,
                    labels=labels,
                    mask=mask,
                    dropout=dropout,
                    token2idx=token2idx,
                    idx2token=idx2token)
    # train(model=m,
    #       pieces=pieces,
    #       token2idx=token2idx,
    #       epochs=500000,
    #       save_name="model/jsb8_30/model_",
    #       load_name="model/jsb8/model_0.0013442965-210500")\
    # train(model=m,
    #       pieces=pieces,
    #       token2idx=token2idx,
    #       epochs=500000,
    #       save_name="model/jsb8_fine/model_",
    #       load_name="model/jsb8/model_0.0013442965-210500",
    #       finetuning=True)
    # test(m, pieces, "model/jsb8_fine/model_0.01460308-15000")
    generate(m, pieces, "model/jsb8_fine/model_0.01460308-15000", token2idx, idx2token)
