import tkinter as tk
import os
import cv2
from PIL import Image, ImageTk
import joblib
import mysql.connector
import mediapipe as mp
import subprocess
import time
from pathlib import Path


from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# Database settings are read from environment variables so credentials are not
# stored in source control. The defaults match a typical local MySQL setup.
databasehost = os.getenv("HERMES_DB_HOST", "localhost")
databaseuser = os.getenv("HERMES_DB_USER", "root")
databasepassword = os.getenv("HERMES_DB_PASSWORD", "")
databasename = os.getenv("HERMES_DB_NAME", "hermes_db")

def resource_path(relpath):
    return Path(__file__).resolve().parent / relpath



window = tk.Tk()

# interface colours and fonts
bgcolor = "#0F172A"
maintextcolor = "#E5E7EB"
secondarytextcolor = "#CBD5E1"
buttoncolor = "#38BDF8"
buttonhovercolor = "#0EA5E9"

bodyfont = ("Rubik", 14)
statusfont = ("Rubik", 16)
buttonfont = ("Montagu Slab", 14, "bold")

window.config(bg=bgcolor)
window.title("HERMES")
window.geometry("1200x800")
appicon = ImageTk.PhotoImage(
    file=str(resource_path("assets/icons/macOS icons/icon_512x512.png"))
)
window.iconphoto(True, appicon)
window.appicon = appicon

titlelabel = tk.Label(
    window,
    text="Project H.E.R.M.E.S.",
    font=("Nose Transport 9", 42, "bold"),
    bg=bgcolor,
    fg=maintextcolor,
    bd=0,
    relief="flat"
)
titlelabel.pack(pady=(18, 4))


subtitlelabel = tk.Label(
    window, text="Hand-sign Extraction and Recognition using Machine-learning for Enhanced Speech",
    font=bodyfont, bg=bgcolor, fg=secondarytextcolor, bd=0, relief="flat"
)
subtitlelabel.pack()
subtitlelabel = tk.Label(
    window, text="Property of Hanin Mohammed",
    font=bodyfont, bg=bgcolor, fg=secondarytextcolor, bd=0, relief="flat"
)
subtitlelabel.pack(pady=(0, 12))

camerastatuslabel = tk.Label(
    window, text="Camera status: Off",
    font=statusfont, bg=bgcolor, fg=maintextcolor, bd=0, relief="flat"
)
camerastatuslabel.pack(pady=(4, 8))
cameralabel = tk.Label(window, bg=bgcolor, bd=0, relief="flat")
cameralabel.pack(pady=(6, 12))


recognisedsignlabel = tk.Label(
    window, text="Recognised sign: None",
    font=statusfont, bg=bgcolor, fg=maintextcolor, bd=0, relief="flat"
)
recognisedsignlabel.pack(pady=5)

matchlabel = tk.Label(
    window, text="Match: None",
    font=statusfont, bg=bgcolor, fg=maintextcolor, bd=0, relief="flat"
)
matchlabel.pack(pady=5)

infolabel = tk.Label(
    window,
    text="No sign selected",
    font=bodyfont,
    bg=bgcolor,
    fg=secondarytextcolor,
    bd=0,
    relief="flat",
    wraplength=720
)
infolabel.pack(pady=5)



camera = None
lastspokensign = ""
pendingsign = ""
pendingstarttime = 0

# model files
knnmodelfile = str(resource_path("models/landmark_knn_model.pkl"))
knnmodel = joblib.load(knnmodelfile)

handmodelfile = str(resource_path("models/hand_landmarker.task"))
handoptions = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=handmodelfile),
    running_mode=VisionTaskRunningMode.IMAGE,
    num_hands=2
)

handdetector = vision.HandLandmarker.create_from_options(handoptions)

handconnections = vision.HandLandmarksConnections.HAND_CONNECTIONS



# camera starting
def start_camera():
    global camera

    if camera is not None:
        camerastatuslabel.config(text="Camera status: Already on")
        return

    camerastatuslabel.config(text="Camera status: Starting")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camerastatuslabel.config(text="Camera status: Error")
        camera = None
        return

    camerastatuslabel.config(text="Camera status: On")

    show_frame()



# frame processing
def show_frame():
    global camera

    if camera is None:
        return

    success, frame = camera.read()
    if not success:
        camerastatuslabel.config(text="Camera status: Read error")
        return

    smallframe = cv2.resize(frame, (640, 360))
    flippedframe = cv2.flip(smallframe, 1)

    landmarkrow = draw_hand_tracking(flippedframe)

    if landmarkrow is None:
        update_prediction(
            "No hand detected",
            "None",
            "Please show your hand to the camera."
        )
    else:
        signname, matchvalue, signinformation, speechtext = predict_sign(landmarkrow)
        update_prediction(signname, matchvalue, signinformation)
        speak_sign_if_needed(signname, speechtext)


    rgbframe = cv2.cvtColor(flippedframe, cv2.COLOR_BGR2RGB)

    pilimage = Image.fromarray(rgbframe)
    tkimage = ImageTk.PhotoImage(image=pilimage)

    cameralabel.config(image=tkimage)
    cameralabel.image = tkimage

    window.after(10, show_frame)

# gather data
def draw_hand_tracking(cameraframe):
    frameheight, framewidth = cameraframe.shape[:2]

    rgbframe = cv2.cvtColor(cameraframe, cv2.COLOR_BGR2RGB)

    mediapipeimage = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgbframe
    )

    handresult = handdetector.detect(mediapipeimage)

    if not handresult.hand_landmarks:
        return None

    landmarkrow = []
    firsthand = handresult.hand_landmarks[0]

    for landmark in firsthand:
        landmarkrow.append(landmark.x)
        landmarkrow.append(landmark.y)
        landmarkrow.append(landmark.z)

    for handlandmarks in handresult.hand_landmarks:
        for landmark in handlandmarks:
            xpos = int(landmark.x * framewidth)
            ypos = int(landmark.y * frameheight)

            cv2.circle(cameraframe, (xpos, ypos), 5, (0, 255, 0), -1)

        for connection in handconnections:
            startind = connection.start
            endind = connection.end

            startlandmark = handlandmarks[startind]
            endlandmark = handlandmarks[endind]

            startx = int(startlandmark.x * framewidth)
            starty = int(startlandmark.y * frameheight)

            endx = int(endlandmark.x * framewidth)
            endy = int(endlandmark.y * frameheight)

            cv2.line(cameraframe, (startx, starty), (endx, endy), (0, 0, 255), 2)

    return landmarkrow



# prediction
def predict_sign(landmarkrow):
    distances = knnmodel.kneighbors([landmarkrow], return_distance=True)[0]
    nearestdistance = distances[0][0]

    maxdistance = 0.35

    if nearestdistance > maxdistance:
        signname = "Uncertain"
        signinformation = "Please show the sign more clearly to the camera."
        speechtext = ""
        matchvalue = "Uncertain"
    else:
        prediction = knnmodel.predict([landmarkrow])
        signname = prediction[0]
        signinformation, speechtext = get_sign_information(signname)
        matchvalue = "Accepted"

    return signname, matchvalue, signinformation, speechtext

# retrieve the recognised sign's information and speech from MySQL
def get_sign_information(signname):
    try:
        database = mysql.connector.connect(
            host=databasehost,
            user=databaseuser,
            password=databasepassword,
            database=databasename
        )
    except mysql.connector.Error:
        return "Error connecting to database.", ""

    cursor = database.cursor()
    query = "Select sign_information, speech_text from signs where sign_name=%s"
    values = (signname,)
    cursor.execute(query, values)
    result = cursor.fetchone()
    cursor.close()
    database.close()

    if result is None:
        return "Information not found in database.", ""

    return result[0], result[1]



# display the result in the interface
def update_prediction(signname, matchvalue, signinformation):
    recognisedsignlabel.config(text="Recognised sign: " + signname)
    matchlabel.config(text="Match: " + matchvalue)
    infolabel.config(text=signinformation)

# wait for a stable new sign before speaking
def speak_sign_if_needed(signname, speechtext):
    global lastspokensign, pendingsign, pendingstarttime

    if signname == "Uncertain":
        return

    if signname == "No hand detected":
        return

    if signname != pendingsign:
        pendingsign = signname
        pendingstarttime = time.time()
        return

    if time.time() - pendingstarttime >= 0.4:
        if signname != lastspokensign:
            speak_text(speechtext)
            lastspokensign = signname


# speak without stopping the camera updates
def speak_text(words):
    subprocess.Popen(["say", words])



# stop and release the camera
def stop_camera():
    global camera

    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None
        camerastatuslabel.config(text="Camera status: Off")
        cameralabel.config(image="")
        cameralabel.image = None
    else:
        camerastatuslabel.config(text="Camera status: Already off")


startbutton = tk.Button(
    window, text="Start Camera", font=buttonfont,
    bg=buttoncolor, fg="#020617",
    activebackground=buttonhovercolor, activeforeground="#020617",
    bd=0, relief="flat", padx=18, pady=8, command=start_camera
)
startbutton.pack(pady=(14, 6))

stopbutton = tk.Button(
    window, text="Stop Camera", font=buttonfont,
    bg=buttoncolor, fg="#020617",
    activebackground=buttonhovercolor, activeforeground="#020617",
    bd=0, relief="flat", padx=18, pady=8, command=stop_camera
)
stopbutton.pack(pady=6)

window.mainloop()
