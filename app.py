from flask import Flask, json, Response, request, render_template
from werkzeug.utils import secure_filename
from os import path, getcwd
import time
from db import Database
from face import Face,realtimeTrain
from PIL import Image
import shutil
import face_recognition_api
from annoy import AnnoyIndex


app = Flask(__name__)

app.config['file_allowed'] = ['image/png', 'image/jpeg']
app.config['storage'] = path.join(getcwd(), 'storage')
app.db = Database()
app.face = Face(app)


def face_gen(image_path,face_output):
    # Load the jpg file into a numpy array
    image = face_recognition_api.load_image_file(image_path)

    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.
    # See also: find_faces_in_picture.py
    face_locations = face_recognition_api.face_locations(image)#face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    
    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    if len(face_locations)<1:
        outname=face_output.split('/')[-1]
        shutil.copy(image_path, 'trainData2face/zero_face/'+outname)

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(face_output)
        #pil_image.show()
def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


def error_handle(error_message, status=500, mimetype='application/json'):
    return Response(json.dumps({"error": {"message": error_message}}), status=status, mimetype=mimetype)


def get_user_by_id(user_id):
    user = {}
    results = app.db.select(
        'SELECT users.id, users.name, users.created, faces.id, faces.user_id, faces.filename,faces.created FROM users LEFT JOIN faces ON faces.user_id = users.id WHERE users.id = ?',
        [user_id])

    index = 0
    for row in results:
        # print(row)
        face = {
            "id": row[3],
            "user_id": row[4],
            "filename": row[5],
            "created": row[6],
        }
        if index == 0:
            user = {
                "id": row[0],
                "name": row[1],
                "created": row[2],
                "faces": [],
            }
        if row[3]:
            user["faces"].append(face)
        index = index + 1

    if 'id' in user:
        return user
    return None


def delete_user_by_id(user_id):
    app.db.delete('DELETE FROM users WHERE users.id = ?', [user_id])
    # also delete all faces with user id
    app.db.delete('DELETE FROM faces WHERE faces.user_id = ?', [user_id])

#   Route for Hompage
@app.route('/', methods=['GET'])
def page_home():

    return render_template('index.html')

@app.route('/api', methods=['GET'])
def homepage():
    output = json.dumps({"api": '1.0'})
    return success_handle(output)


@app.route('/api/train', methods=['POST'])
def train():
    output = json.dumps({"success": True})

    if 'file' not in request.files:

        print ("Face image is required")
        return error_handle("Face image is required.")
    else:

        print("File request", request.files)
        file = request.files['file']

        if file.mimetype not in app.config['file_allowed']:

            print("File extension is not allowed")

            return error_handle("We are only allow upload file with *.png , *.jpg")
        else:

            # get name in form data
            name = request.form['name']

            print("Information of that face", name)

            print("File is allowed and will be saved in ", app.config['storage'])
            filename = secure_filename(file.filename)
            trained_storage = path.join(app.config['storage'], 'trained')
            untrained_storage = path.join(app.config['storage'], 'untrained')
            
            file.save(path.join(untrained_storage, filename))
            face_gen(path.join(untrained_storage, filename),path.join(trained_storage, filename))
            # let start save file to our storage

            # save to our sqlite database.db
            created = int(time.time())
            user_id = app.db.insert('INSERT INTO users(name, created) values(?,?)', [name, created])

            if user_id:

                print("User saved in data", name, user_id)
                # user has been save with user_id and now we need save faces table as well

                face_id = app.db.insert('INSERT INTO faces(user_id, filename, created) values(?,?,?)',
                                        [user_id, filename, created])

                if face_id:

                    print("cool face has been saved")
                    realtimeTrain()
                    face_data = {"id": face_id, "filename": filename, "created": created}
                    return_output = json.dumps({"id": user_id, "name": name, "face": [face_data]})
                    return success_handle(return_output)
                else:

                    print("An error saving face image.")

                    return error_handle("n error saving face image.")

            else:
                print("Something happend")
                return error_handle("An error inserting new user")

        print("Request is contain image")
    
    return success_handle(output)


# route for user profile
@app.route('/api/users/<int:user_id>', methods=['GET', 'DELETE'])
def user_profile(user_id):
    if request.method == 'GET':
        user = get_user_by_id(user_id)
        if user:
            return success_handle(json.dumps(user), 200)
        else:
            return error_handle("User not found", 404)
    if request.method == 'DELETE':
        delete_user_by_id(user_id)
        return success_handle(json.dumps({"deleted": True}))


# router for recognize a unknown face
@app.route('/api/recognize', methods=['POST'])
def recognize():

    if 'file' not in request.files:
        return error_handle("Image is required")
    else:
        file = request.files['file']
        # file extension valiate
        if file.mimetype not in app.config['file_allowed']:
            return error_handle("File extension is not allowed")
        else:

            filename = secure_filename(file.filename)
            unknown_storage = path.join(app.config["storage"], 'unknown')
            unknown_storage_face = path.join(app.config["storage"], 'unknown_face')
            file_path = path.join(unknown_storage, filename)
            file.save(file_path)
            face_gen(file_path,path.join(unknown_storage_face, filename))

            user_id,(pred_age,prob_age),(pred_gender,prob_gender) = app.face.recognize(filename)
            user={}
            if user_id:
                if user_id=='unknown':
                    user['name']='unknown'
                else:
                    user = get_user_by_id(user_id)
                user['name']=user['name']+' Guess@age:'+str(pred_age)+'Guess@gender:'+str(pred_gender)
                message = {"message": "Hey we found {0} matched with your face image".format(user["name"]),
                           "user": user}
                return success_handle(json.dumps(message))
            else:

                return error_handle("Sorry we can not found any people matched with your face image, try another image")


# Run the app
app.run()