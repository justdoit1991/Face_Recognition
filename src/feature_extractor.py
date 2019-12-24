import os 
import re
import numpy as np 
import tensorflow as tf 
from tensorflow.python.platform import gfile

tf.logging.set_verbosity(tf.logging.WARN)

class FeatureExtractModel():
    def __init__(self, model_path, gpu_memory_fractio=0.6):
        self.model_path = model_path
        self.graph = tf.Graph()
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fractio)
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=self.gpu_options))
        with self.sess.as_default():
            with self.graph.as_default():
                self.__load_model()
                self.input_placeholder = self.graph.get_tensor_by_name('input:0')
                self.train_placeholder = self.graph.get_tensor_by_name('phase_train:0')
                self.embeddings = self.graph.get_tensor_by_name('embeddings:0')
                self.prelogits = self.graph.get_tensor_by_name('InceptionResnetV1/Bottleneck/MatMul:0')

    def __load_model(self):
        if os.path.isfile(self.model_path):
            print('** load model name : {}'.format(self.model_path))
            self.model_name = self.model_path
            try:
                with gfile.FastGFile(self.model_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, input_map=None, name='')
            except Exception as e:
                print('*** failed to load model. ***')
                print(e)
        else:
            try:
                print('** load model dir : {}'.format(self.model_path))
                meta_file, ckpt_file = get_model_filename(self.model_path)
                print('  --metagraph  : {}'.format(meta_file))
                print('  --checkpoint : {}'.format(ckpt_file))
                self.saver = tf.train.import_meta_graph(os.path.join(self.model_path, meta_file))
                self.saver.restore(self.sess, os.path.join(self.model_path, ckpt_file))
            except Exception as e:
                print('*** failed to load model. ***')
                print(e)

    def __normalize(self, img):
        return np.multiply(np.subtract(img, 127.5), 1 / 128)

    def infer(self, faces):
        normalized = [self.__normalize(f) for f in faces]
        features = self.sess.run(self.embeddings, feed_dict={self.input_placeholder : normalized, self.train_placeholder : False})
        return features

    def get_prelogits(self, faces):
        normalized = [self.__normalize(f) for f in faces]
        # normalized = [f for f in faces]
        features = self.sess.run(self.prelogits, feed_dict={self.input_placeholder : normalized, self.train_placeholder : False})
        return features

def get_model_filename(model_dir):
    """
    Given a model dir, return meta graph and checkpoint filename
    """
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file