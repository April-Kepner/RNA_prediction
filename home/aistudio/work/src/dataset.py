import os
from collections import defaultdict

def read_data(filename, test=False):
    data, x = [], []
    for line in open(filename, "r"):
        line = line.strip()
        if not line:
            ID, seq, dot = x[:3]
            if test:
                x = {"id": ID,
                     "sequence": seq,
                     "structure": dot,
                }
                data.append(x)
                x = []
                continue
            punp = x[3:]
            punp = [punp_line.split() for punp_line in punp]
            punp = [(float(p)) for i, p in punp]
            x = {"id": ID,
                 "sequence": seq,
                 "structure": dot,
                 "p_unpaired": punp,
            }
            data.append(x)
            x = []
        else:
            x.append(line)
    return data

def load_train_data():
    assert os.path.exists("work/data/train.txt")
    assert os.path.exists("work/data/dev.txt")
    train = read_data("work/data/train.txt")
    dev = read_data("work/data/dev.txt")
    return train, dev

def load_test_data():
    assert os.path.exists("work/data/test_nolabel.txt")
    test = read_data("work/data/test_nolabel.txt", test=True)
    return test

def load_test_label_data():
    assert os.path.exists("work/data/dev.txt")
    test = read_data("work/data/dev.txt")
    return test


