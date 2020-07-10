import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
from functools import partial

def read_npy_file(split, feat_source, tshape, item):
    data = np.load('/data/taiyin/vizwiz/{}_feat/{}/{}.npy'.format(feat_source, split, item.decode()[:-4]))
    data = data.reshape(tshape)
    return data.astype(np.float32)

def set_feat_shape(tshape, item):
    item.set_shape(tshape)
    return item

def data_iterator(split, feat_source, batch_size):
    my_data = json.load(open('data/quality.json'))[split]
    tshape = (10, 10, 2048) if feat_source == 'detectron' else (14, 14, 2048)
    ds_feat = tf.data.Dataset.from_tensor_slices(my_data['image'])
    ds_feat = ds_feat.map(lambda item: tf.py_func(partial(read_npy_file, split, feat_source, tshape), [item], tf.float32))
    ds_feat = ds_feat.map(partial(set_feat_shape, tshape))
    ds_rec = tf.data.Dataset.from_tensor_slices(my_data['recognizable'])
    ds = tf.data.Dataset.zip((ds_feat, ds_rec))    
    if split == 'train':
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size=256)
    ds = ds.prefetch(2)
    iterator = tf.compat.v1.data.make_initializable_iterator(ds)

    return iterator