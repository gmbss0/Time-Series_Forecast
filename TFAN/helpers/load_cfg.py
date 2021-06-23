import os

def load_cfg(filename):
    cfg = {}
    with open(os.path.join(os.getcwd(),filename)) as f:
        for line in f:
            (key, value) = line.split()
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
            cfg[key] = value
    return cfg
        
cfg = load_cfg("cfg.txt")