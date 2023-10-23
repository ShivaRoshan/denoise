import imageio
import os
import numpy as np
import tensorflow as tf
import re
import math

def imread(path):
    img=imageio.imread(path).astype(np.float32)
    img=img/255.
    return img

def load(saver, sess, checkpoint_dir, folder):
    print('==================== Reading Checkpoints ====================')
    checkpoint = os.path.join(checkpoint_dir, folder)

    ckpt= tf.train.get_checkpoint_state(checkpoint)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint, ckpt_name))

        step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

        print("=================== Success to load {} ====================".format(ckpt_name))
        return True, step
    else:
        print("=================== Fail to find a Checkpoint ====================")
        return False, 0

def save(saver, sess, checkpoint_dir, trial, step):
    model_name='model'
    checkpoint = os.path.join(checkpoint_dir, 'Model%d'% trial)

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    saver.save(sess, os.path.join(checkpoint, model_name), global_step=step)

def count_param(scope=None):
    N=np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)])
    print('Model Params: %d K' % (N/1000))

def psnr(img1, img2):
    img1=np.float32(img1)
    img2=np.float32(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX= 1.0
    else:
        PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
