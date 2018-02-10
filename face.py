import face_recognition
from os import path, getcwd
from gender_age import PredictClass
from db import Database
import face_recognition_api
from annoy import AnnoyIndex
import tensorflow as tf

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ntrees = 100
metric = 'angular'  # or 'euclidean'
description = 'image classification'

try:
    import cPickle as pickle    #python 2
except ImportError as e:
    import pickle   #python 3
import time
#tl1=time.time()
#s = shelve.open('names.bin')
#u = AnnoyIndex(s['layer_size'],metric)
#u.load('index.ann')
#u=load_model('index.ann')
#tl2=time.time()
#print('loading model time:%f s'%(tl2-tl1))
config= tf.ConfigProto(allow_soft_placement=True)
gender_model_dir='models/gender/inception/21936'
age_model_dir='models/age/inception/22801'

age_predict=PredictClass(config,'age',age_model_dir)
gender_predict=PredictClass(config,'gender',gender_model_dir)


def find_matching_id(embedding,annoy_tree):
    threshold = 0.4
    min_dist = 10.0
    matching_id = None
    near=annoy_tree.get_nns_by_vector(embedding,1,include_distances=True)
    dist=near[1][0]
    if dist < threshold and dist < min_dist:      
        #matching_id = unicode(s[str(near[0][0])],'utf-8','ignore')
        matching_id=near[0][0]
        p=round(float(dist),4)
        prob=unicode(p)
        #prob=unicode(str(p),'utf-8','ignore')
        min_dist = prob
        return matching_id, min_dist
    else:
        return None
    
class realtimeTrain:
    def __init__(self):
        self.storage = path.join(getcwd(), 'storage')
        self.db = Database()
        self.faces = []  # storage all faces in caches array of face object
        self.known_encoding_faces = []  # faces data for recognition
        self.face_user_keys = {}
        self.load_all()

    def load_user_by_index_key(self, index_key=0):

        key_str = str(index_key)

        if key_str in self.face_user_keys:
            return self.face_user_keys[key_str]

        return None

    def load_train_file_by_name(self, name):
        trained_storage = path.join(self.storage, 'trained')
        return path.join(trained_storage, name)

    def load_unknown_file_by_name(self, name):
        unknown_storage = path.join(self.storage, 'unknown')
        return path.join(unknown_storage, name)

    def load_all(self):

        results = self.db.select('SELECT faces.id, faces.user_id, faces.filename, faces.created FROM faces')
        self.layer_size=0
        count=0
        for row in results:

            user_id = row[1]
            filename = row[2]
            print('train::',user_id)
            face = {
                "id": row[0],
                "user_id": user_id,
                "filename": filename,
                "created": row[3]
            }
            self.faces.append(face)

            face_image = face_recognition_api.load_image_file(self.load_train_file_by_name(filename))
            face_image_encoding = face_recognition_api.face_encodings(face_image)[0]
            index_key = len(self.known_encoding_faces)
            self.known_encoding_faces.append(face_image_encoding)
            index_key_string = str(index_key)
            self.face_user_keys['{0}'.format(index_key_string)] = user_id
            if count==0:
                self.layer_size=len(face_image_encoding)
                self.tree = AnnoyIndex(self.layer_size,metric) # prepare index
            self.tree.add_item(user_id,face_image_encoding)
            count+=1
        print 'building index...\n'
        if self.layer_size>0:
            print 'layer_size=',self.layer_size
            self.tree.build(ntrees)
            self.tree.save('index.ann')


def loadannoy():
    tl1=time.time()
    metric='angular' 
    layer_size=128
    annoytree = AnnoyIndex(layer_size,metric)
    annoytree.load('index.ann')
    tl2=time.time()
    print('loading model time:%f s'%(tl2-tl1))
    return annoytree
class Face:
    def __init__(self, app):
        self.storage = app.config["storage"]
        self.db = app.db
        self.faces = []  # storage all faces in caches array of face object
        self.known_encoding_faces = []  # faces data for recognition
        self.face_user_keys = {}
        self.load_all()

    def load_user_by_index_key(self, index_key=0):

        key_str = str(index_key)

        if key_str in self.face_user_keys:
            return self.face_user_keys[key_str]

        return None

    def load_train_file_by_name(self, name):
        trained_storage = path.join(self.storage, 'trained')
        return path.join(trained_storage, name)

    def load_unknown_file_by_name(self, name):
        unknown_storage = path.join(self.storage, 'unknown')
        unknown_storage_face = path.join(self.storage, 'unknown_face')
        return (path.join(unknown_storage, name),path.join(unknown_storage_face, name))

    def load_all(self):

        results = self.db.select('SELECT faces.id, faces.user_id, faces.filename, faces.created FROM faces')
        self.layer_size=0
        count=0
        for row in results:

            user_id = row[1]
            filename = row[2]
            
            face = {
                "id": row[0],
                "user_id": user_id,
                "filename": filename,
                "created": row[3]
            }
            self.faces.append(face)

            face_image = face_recognition_api.load_image_file(self.load_train_file_by_name(filename))
            face_image_encoding = face_recognition_api.face_encodings(face_image)[0]
            index_key = len(self.known_encoding_faces)
            self.known_encoding_faces.append(face_image_encoding)
            index_key_string = str(index_key)
            self.face_user_keys['{0}'.format(index_key_string)] = user_id
            print('user_id',user_id)
            if count==0:
                self.layer_size=len(face_image_encoding)
                self.tree = AnnoyIndex(self.layer_size,metric) # prepare index
            self.tree.add_item(user_id,face_image_encoding)
            count+=1
        print 'building index...\n'
        if self.layer_size>0:
            print 'layer_size=',self.layer_size
            self.tree.build(ntrees)
            self.tree.save('index.ann')


    def recognize(self, unknown_filename):
        tree=loadannoy()
        (unfile,unfile_face)=self.load_unknown_file_by_name(unknown_filename)
        
        unknown_image = face_recognition_api.load_image_file(unfile)
        unknown_encoding_image = face_recognition_api.face_encodings(unknown_image)[0]

        #results = face_recognition.compare_faces(self.known_encoding_faces, unknown_encoding_image);
        results2=find_matching_id(unknown_encoding_image,tree)
        guess_age=age_predict.predict([unfile_face])
        guess_gender=gender_predict.predict([unfile_face])
        #print("results", results)
        print("results2", results2)
        if results2:
            matching_id, min_dist=results2
            user_id=matching_id#self.load_user_by_index_key(matching_id)
            return (user_id,guess_age,guess_gender)
        
        return ('unknown',guess_age,guess_gender)
        '''
        index_key = 0
        for matched in results:

            if matched:
                # so we found this user with index key and find him
                user_id = self.load_user_by_index_key(index_key)

                return user_id

            index_key = index_key + 1

        return None
        '''
