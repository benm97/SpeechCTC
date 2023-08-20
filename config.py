from bidict import bidict
LAS_VOCAB = bidict({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6,
                           "g": 7, "h": 8, "i": 9, "j": 10, "k": 11, "l": 12, "m": 13,
                           "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20,
                           "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26,
                           " ": 27, "<eos>": 28})
N_MFCC = 39
SEQUENCE_LENGTH_MAX = 268
MFCC_MAX_FRAME = 300

# training params
batch_size = 32
learning_rate = 0.005