import numpy as np
import pickle
import hyper_params as hp
import random


def load_pieces(dirpath):
    '''
    Loads pianoroll into numpy matrices.
    dirpath: string, the path to the pianoroll.
    Returns
    pieces: dictionary with keys ("train", "test", "valid"). Each value is
    a numpy array of size (pieces_num, window_size + 1)
    seqlens: dictionary with keys ("train", "test", "valid"). Each value is
    a numpy array of size (pieces_num), with each value being the length of
    of the piece unpadded.
    '''
    file = open(dirpath, "rb")
    pieces = pickle.load(file, encoding="latin1")
    pieces = clean_pieces(pieces)
    pieces, seqlens = pad_pieces_to_max(pieces)
    return pieces, seqlens


def clean_pieces(pieces):
    '''
    Pads chords to four voices and serialize the pieces.
    pieces: dictionary with keys ("train", "test", "valid"). Each value is
    a list of size (pieces_num, piece_length, chord_size)
    Returns
    pieces: dictionary with keys ("train", "test", "valid"). Each value is
    a numpy array of size (pieces_num, piece_length * 4)
    '''
    def pad(chord):
        # pad to 4 voices
        padded = np.array(list(chord))
        while len(padded) < 4:
            padded = np.append([hp.REST], padded)
        return padded

    def clean_piece(piece):
        # pad and serialize
        return np.array([pad(chord) for chord in piece]).flatten()
    pieces["train"] = np.array([clean_piece(p) for p in pieces["train"]])
    pieces["test"] = np.array([clean_piece(p) for p in pieces["test"]])
    pieces["valid"] = np.array([clean_piece(p) for p in pieces["valid"]])
    return pieces


def pad_pieces_to_max(pieces):
    '''
    Pad each piece up to the maximum length.
    pieces: dictionary with keys ("train", "test", "valid"). Each value is
    a numpy array of size (pieces_num, piece_length * 4)
    Returns
    pieces: dictionary with keys ("train", "test", "valid"). Each value is
    a numpy array of size (pieces_num, window_size + 1)
    seqlens: dictionary with keys ("train", "test", "valid"). Each value is
    a numpy array of size (pieces_num), with each value being the length of
    of the piece unpadded.
    '''
    def pad_piece_to_max(piece):
        while len(piece) < hp.MAX_LEN + 1:
            piece = np.append(piece, [hp.PAD])
        return piece

    def seperate_long_piece(pieces_list):
        # pieces longer than max length are chopped into multiple pieces
        # shorter pieces are padded up to max length
        new_pieces = []
        for i in range(len(pieces_list)):
            piece = np.append(pieces_list[i], [hp.STOP])
            if len(pieces_list[i]) > hp.MAX_LEN + 1:
                new_pieces = new_pieces \
                             + [piece[j:j+hp.MAX_LEN+1] for j in
                                range(0, len(piece), hp.SEPERATION)]
            else:
                new_pieces.append(pad_piece_to_max(piece))
        new_pieces = list(
            filter(lambda x: len(x) == hp.MAX_LEN + 1, new_pieces)
        )
        return np.array(new_pieces)

    pieces["train"] = seperate_long_piece(pieces["train"])
    pieces["test"] = seperate_long_piece(pieces["test"])
    pieces["valid"] = seperate_long_piece(pieces["valid"])

    seqlens = {
        "train": np.zeros(len(pieces["train"])),
        "test": np.zeros(len(pieces["test"])),
        "valid": np.zeros(len(pieces["valid"]))
    }

    for i in range(len(pieces["train"])):
        seqlens["train"][i] = len(pieces["train"][i])
    for i in range(len(pieces["test"])):
        seqlens["test"][i] = len(pieces["test"][i])
    for i in range(len(pieces["valid"])):
        seqlens["valid"][i] = len(pieces["valid"][i])
    return pieces, seqlens


def build_vocab(pieces):
    '''
    Build vocab from the pieces, creates the vocab dictionary.
    pieces: dictionary with keys ("train", "test", "valid"). Each value is
    a numpy array of size (pieces_num, window_size + 1)
    Returns
    token2idx: dictionary. With keys being the notes, and values being the
    indices.
    idx2token: dictionary. With keys being the indices, and values being the
    notes.
    '''
    total_notes = np.hstack((pieces["train"].astype(int).flatten(),
                             pieces["test"].astype(int).flatten(),
                             pieces["valid"].astype(int).flatten()))
    vocabs = set(total_notes)
    if hp.PAD in vocabs:
        vocabs.remove(hp.PAD)
    if hp.MASK in vocabs:
        vocabs.remove(hp.MASK)
    idx2token = {i+1: w for i, w in enumerate(vocabs)}
    token2idx = {w: i+1 for i, w in enumerate(vocabs)}
    idx2token[0] = hp.PAD
    idx2token[1] = hp.MASK
    token2idx[hp.PAD] = 0
    token2idx[hp.MASK] = 1

    return token2idx, idx2token


def tokenize(pieces, token2idx, idx2token):
    '''
    Translates notes in pieces to indices using provided vocab dictionaries.
    pieces: dictionary with keys ("train", "test", "valid"). Each value is
    a numpy array of size (pieces_num, window_size + 1)
    token2idx: dictionary. With keys being the notes, and values being the
    indices.
    idx2token: dictionary. With keys being the indices, and values being the
    notes.
    Returns
    pieces: dictionary with keys ("train", "test", "valid"). Each value is
    a numpy array of size (pieces_num, window_size + 1), with each value a
    translated index.
    '''
    pieces["train"] = np.array(
        [[token2idx[w] for w in p] for p in pieces["train"]]
    )
    pieces["test"] = np.array(
        [[token2idx[w] for w in p] for p in pieces["test"]]
    )
    pieces["valid"] = np.array(
        [[token2idx[w] for w in p] for p in pieces["valid"]]
    )
    return pieces


def get_finetuning_batch(pieces,
                         token2idx,
                         batch_size=hp.BATCH_SIZE,
                         testing=False):
    '''
    For language model finetuning.
    Gets a randomized batch of batch size number of pieces from pieces.
    Where the input has all values that's not the soprano masked.
    pieces: dictionary with keys ("train", "test", "valid"). Each value is
    a numpy array of size (pieces_num, window_size + 1)
    token2idx: dict, translates tokens to indices.
    batch_size: int, default hp.BATCH_SIZE
    testing: bool, default False
    Returns
    x: a numpy array of size (pieces_num, window_size)
    y: a numpy array of size (pieces_num, window_size)
    mask: a numpy array of size (pieces_num, window_size)
    '''
    if testing:
        tr_te = "test"
    else:
        tr_te = "train"
    batch_indices = np.random.choice(len(pieces[tr_te]),
                                     size=batch_size,
                                     replace=True)
    x = pieces[tr_te][batch_indices][:, :-1]
    y = np.copy(x)
    mask = np.zeros_like(x)

    for i in range(len(x)):
        for j in range(0, len(x[i]), 4):
            x[i][j+1] = token2idx[hp.MASK]  # alto
            mask[i][j+1] = 1
            x[i][j+2] = token2idx[hp.MASK]  # tenor
            mask[i][j+2] = 1
            x[i][j+3] = token2idx[hp.MASK]  # bass
            mask[i][j+3] = 1

    return x.astype(int), y.astype(int), mask.astype(float)


def get_pretraining_batch(pieces,
                          token2idx,
                          batch_size=hp.BATCH_SIZE,
                          mask_prob=0.15,
                          testing=False):
    '''
    For language model pretraining.
    Gets a randomized batch of batch size number of pieces from pieces.
    Where the input has a mask_prob portion of its values masked, in a
    80-10-10 scheme detailed in the BERT paper.
    pieces: dictionary with keys ("train", "test", "valid"). Each value is
    a numpy array of size (pieces_num, window_size + 1)
    token2idx: dict, translates tokens to indices.
    batch_size: int, default hp.BATCH_SIZE
    mask_prob: float, default 0.15
    testing: bool, default False
    Returns
    x: a numpy array of size (pieces_num, window_size)
    y: a numpy array of size (pieces_num, window_size)
    mask: a numpy array of size (pieces_num, window_size)
    '''
    if testing:
        tr_te = "test"
    else:
        tr_te = "train"
    batch_indices = np.random.choice(len(pieces[tr_te]),
                                     size=batch_size,
                                     replace=True)
    x = pieces[tr_te][batch_indices][:, :-1]
    y = np.copy(x)
    mask = np.zeros_like(x)

    for i in range(len(x)):
        mask_indices = np.random.choice(len(x[i]),
                                        size=int(len(x[i]) * mask_prob),
                                        replace=False)
        print(len(mask_indices))
        for j in mask_indices:
            r = random.random()
            mask[i][j] = 1
            if r <= .1:
                continue
            elif r <= .2:
                random_token = random.randint(0, len(token2idx) - 1)
                x[i][j] = random_token
            else:
                x[i][j] = token2idx[hp.MASK]

    return x.astype(int), y.astype(int), mask.astype(float)


# pieces, seqlens = load_pieces("data/roll/jsb8.pkl")
# print(pieces["test"].shape)
# get_batch(pieces)
# token2idx, idx2token = build_vocab(load_pieces("data/roll/jsb8.pkl")[0])
# print(idx2token)
# print(len(token2idx))
# print(tokenize(load_pieces("data/roll/jsb16.pkl")[0], token2idx, idx2token)["train"])
