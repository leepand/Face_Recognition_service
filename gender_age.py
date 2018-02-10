# -*- coding:utf-8 -*-
#封装版本---基于图片的预测可运行
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import json
import csv
from dlibdetect import *
from yolodetect import *
'''
param
'''
RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128

'''
config info
'''
class_type='gender'
model_type='inception'
gender_model_dir='models/gender/inception/21936'
age_model_dir='models/age/inception/22801'
face_detection_type='dlib'
face_detection_model_dir='models/shape_predictor_68_face_landmarks.dat'
device_id='/cpu:0'
checkpoint='checkpoint'
filename = 'storage/unknow/liangning7.jpg'
single_look=False
requested_step=''
target=''



def face_detection_model(face_detection_type, face_detection_model_dir):
    if face_detection_type=='dlib':
        return FaceDetectorDlib(model_name=face_detection_model_dir) 

def one_of(fname, types):
    return any([fname.endswith('.' + ty) for ty in types])

def resolve_file(fname):
    if os.path.exists(fname): return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None


def classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_files, writer):
    try:
        num_batches = math.ceil(len(image_files) / MAX_BATCH_SZ)
        pg = ProgressBar(num_batches)
        for j in range(num_batches):
            start_offset = j * MAX_BATCH_SZ
            end_offset = min((j + 1) * MAX_BATCH_SZ, len(image_files))
            
            batch_image_files = image_files[start_offset:end_offset]
            print(start_offset, end_offset, len(batch_image_files))
            image_batch = make_multi_image_batch(batch_image_files, coder)
            batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
            batch_sz = batch_results.shape[0]
            for i in range(batch_sz):
                output_i = batch_results[i]
                best_i = np.argmax(output_i)
                best_choice = (label_list[best_i], output_i[best_i])
                print('Guess @ 1 %s, prob = %.2f' % best_choice)
                if writer is not None:
                    f = batch_image_files[i]
                    writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))
            pg.update()
        pg.done()
    except Exception as e:
        print(e)
        print('Failed to run all images')

def classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_file, writer):
    try:

        print('Running file %s' % image_file)
        image_batch = make_multi_crop_batch(image_file, coder)

        batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]
    
        for i in range(1, batch_sz):
            output = output + batch_results[i]
        
        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        print('Guess @ 1 %s, prob = %.2f' % best_choice)
        return best_choice 
        nlabels = len(label_list)
        if nlabels > 2:
            output[best] = 0
            second_best = np.argmax(output)
            print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

        if writer is not None:
            writer.writerow((image_file, best_choice[0], '%.2f' % best_choice[1]))
    except Exception as e:
        print(e)
        print('Failed to run image %s ' % image_file)

def list_images(srcfile):
    with open(srcfile, 'r') as csvfile:
        delim = ',' if srcfile.endswith('.csv') else '\t'
        reader = csv.reader(csvfile, delimiter=delim)
        if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
            print('skipping header')
            _ = next(reader)
        
        return [row[0] for row in reader]

face_detection_model=False
files = []
    
if face_detection_model:
    print('Using face detector (%s) %s' % (face_detection_type, face_detection_model_dir))
    face_detect = face_detection_model(face_detection_type, face_detection_model_dir)
    face_files, rectangles = face_detect.run(filename)
    print('rectangles:',face_files,rectangles)
    files += face_files



config = tf.ConfigProto(allow_soft_placement=True)
class PredictClass:
    def __init__(self,session_conf,class_type,model_dir,requested_step='',model_type='inception'):
        self.graph=tf.Graph()#为每个类(实例)单独创建一个graph
        self.session_conf=session_conf
        self.sess=tf.Session(graph=self.graph,config=self.session_conf)#创建新的sess
        self.label_list = AGE_LIST if class_type == 'age' else GENDER_LIST
        self.model_dir=model_dir
        self.requested_step=requested_step
        with self.sess.as_default():
             with self.graph.as_default():                    
                    nlabels = len(self.label_list)
                    print('Executing on %s' % device_id)
                    model_fn = select_model(model_type)
                    with tf.device(device_id):
                        self.images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
                        logits = model_fn(nlabels, self.images, 1, False)
                        init = tf.global_variables_initializer()
                        requested_step = self.requested_step if self.requested_step else None
                        checkpoint_path = '%s' % (self.model_dir)
                        ######gender_model_dir
                        model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, checkpoint)                                   
                        self.saver = tf.train.Saver()
                        self.saver.restore(self.sess, model_checkpoint_path)#从恢复点恢复参数                       
                        self.softmax_output = tf.nn.softmax(logits)
                        self.coder = ImageCoder()

    def predict(self,files):
        with self.sess.as_default():
             with self.graph.as_default():  
                    with tf.device(device_id):
                        # Support a batch mode if no face detection model
                        if len(files) == 0:
                            if (os.path.isdir(filename)):
                                for relpath in os.listdir(filename):
                                    abspath = os.path.join(filename, relpath)
                        
                                    if os.path.isfile(abspath) and any([abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                                        print(abspath)
                                        files.append(abspath)
                            else:
                                files.append(filename)
                                # If it happens to be a list file, read the list and clobber the files
                                if any([filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
                                    files = list_images(FLAGS.filename)
                        writer = None
                        output = None
                        if target:
                            print('Creating output file %s' % target)
                            output = open(target, 'w')
                            writer = csv.writer(output)
                            writer.writerow(('file', 'label', 'score'))
                        image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
                        print(image_files)
                        if single_look:
                            classify_many_single_crop(self.sess, self.label_list, self.softmax_output, self.coder, self.images, image_files, writer)

                        else:
                            for image_file in image_files:
                                result=classify_one_multi_crop(self.sess, self.label_list, self.softmax_output, self.coder, self.images, image_file, writer)
                        return result

#gender_predict=PredictClass(config,'gender',gender_model_dir)
#gender_predict.predict(files)