
import os
from collections import defaultdict, Counter
import pickle
import pandas as pd

SUBMIT_PATH = ''
SIGFIGS = 6

def load_labels():
    label = pd.read_csv('./input/label.csv')
    labelName = label['name'].tolist()
    labelId = label['id'].tolist()
    dict_label = {}
    for (name, id) in zip(labelName, labelId):
        dict_label[name.replace(' ','_')] = int(id)
    return dict_label
dict_label = load_labels()
id_name_label = {}
for item in dict_label.items():
    k, v = item
    id_name_label[int(v)] = k
def read_models(model_weights, blend=None):
    if not blend:
        blend = defaultdict(Counter)
    for m, w in model_weights.items():
        print(m, w)
        with open(os.path.join(SUBMIT_PATH, m ), 'r') as f:
            f.readline()
            for l in f:
                id, r = l.split(',')
                id, r = id, r.split(' ')
                n = len(r)//2 * 2
                for i in range(0, n, 2):
                    k = int(dict_label[r[i]])
                    v = int(10**(SIGFIGS - 1) * float(r[i+1]))
                    blend[id][k] += w * v
    return blend


def write_models(blend, file_name, total_weight):
    with open(os.path.join(SUBMIT_PATH, file_name + '.csv'), 'w') as f:
        f.write('key_id,word\n')
        for id, v in blend.items():
            l = ' '.join(['{}'.format(id_name_label[int(t[0])]) for i_t, t in enumerate(v.most_common(20)) if i_t < 3])
            f.write(','.join([str(id), l + '\n']))
    return None

model_pred = {
    'result/xception/result_191000.csv': 1,
    'result/xception/result_189000.csv': 1,
    'result/xception/result_192000.csv': 1,
                 }

avg = read_models(model_pred)
write_models(avg, 'xception', sum(model_pred.values()))
