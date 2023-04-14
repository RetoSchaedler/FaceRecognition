import face_recognition
import cv2
import numpy as np
import glob

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)


known_face_encodings=[]
known_face_names=[]

# Scan Folder for Reference Faces. The jpg file name is used as face name.
for pic in glob.glob('./refFaces/*.jpg'):
    name=pic.replace('/','\\').split('\\')[-1][:-4]
    image=(face_recognition.load_image_file(pic))
    known_face_encodings.append(face_recognition.face_encodings(image)[0])
    known_face_names.append(name)
    print(name)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    #frame=cv2.imread('IMG_7359.JPG')
    #frame=cv2.resize(frame,(1560,1040),interpolation=cv2.INTER_AREA)

    if True: # process_this_frame:
        small_frame = frame
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)# , model='cnn'
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
