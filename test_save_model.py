import tensorflow as tf
from os import path as osp
import shutil
import joblib


sess = tf.Session()
a = tf.constant(1)
b = tf.Variable(2)
c = tf.add(a, b)
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(c))

inputs = {'a':a, 'b':b}
outputs={'c':c}
saver_elements = dict(session=sess, inputs=inputs, outputs=outputs)
saver_info = {'inputs': {k:v.name for k,v in inputs.items()}, 'outputs': {k:v.name for k,v in outputs.items()}}

fpath = 'tmp_log'
if osp.exists(fpath):
    # simple_save refuses to be useful if fpath already exists,
    # so just delete fpath if it's there.
    shutil.rmtree(fpath)
tf.saved_model.simple_save(export_dir=fpath, **saver_elements)
joblib.dump(saver_info, osp.join(fpath, 'model_info.pkl'))

sess.close()
