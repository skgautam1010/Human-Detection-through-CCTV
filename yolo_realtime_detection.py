import cv2
import numpy as np
import time
from datetime import datetime
import smtplib
import imghdr
import time,re
from email.message import EmailMessage
# importing the client from the twilio
from twilio.rest import Client


# Loading video
#rtsp_link=input("Enter the RTSP Link: ")
#cap = cv2.VideoCapture(rtsp_link)
#rtsp://admin:admin@123@192.168.1.108:554


def sendsms():

    account_sid = 'AC6dd4ea3d0205cebdb3af3a78692f7025'
    auth_token = 'b07b4d4cfd698f19f5ba03d719d58a5e'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
                                body='Alert! Someone has intruded, Please reach out to the place as soon as possible and take necessary actions. For more details please check the mail.',
                                from_='+19472253236',
                                status_callback='http://postb.in/1234abcd',
                                to='+917992346306'
                            )

    #print(message.sid)
    print("SMS Sent!")


    
def sendmail():
    Sender_Email = "sonukumargautam83614@gmail.com"
    Reciever_Email = "sonukumargautamsingh@gmail.com"
    #Password = input('Enter your email account password: ')
    timestr = time.strftime("%y-%m-%d-%H-%M-%S")
    phoneNumRegex = re.compile(r'(\d\d-\d\d-\d\d)-(\d\d-\d\d-\d\d)')
    mo = phoneNumRegex.search(timestr)
    path=mo.group(1)
    newMessage = EmailMessage()                         
    newMessage['Subject'] = "Alert Alert!" 
    newMessage['From'] = Sender_Email
    newMessage['To'] = Reciever_Email 
    newMessage.set_content('Someone Intruded in the restricted area!\nPlease check the attached screenshot and take the necessary action immediately.') 
    new='images/frames.jpg'
    with open(new, 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name

    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(Sender_Email, 'qxjqhfuivrtciwli')              
        smtp.send_message(newMessage)
   

def detect_human():
    
    rtsp_link="rtsp://admin:admin@123@192.168.1.108:554"
    cap = cv2.VideoCapture(rtsp_link)

    net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
    classes = []

        # Load coco
    with open("lib/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))


    face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_fullbody.xml")

        #capturing video
    frame_size=(int(cap.get(3)),int(cap.get(4)))
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")

    detecting_face=False
    detection_stopped_time=None
    timer_started=False
    Seconds_to_record_after_detection=2
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    #frame_id = 0
    current_time=datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    output_video=cv2.VideoWriter(f"videos/{current_time}.mp4",fourcc,20,frame_size)

    #start time and end time for the intrusion detection system
    start_time=input("Enter the Start Time for intrusion in following format (HH:MM:SS): ")
    end_time=input("Enter the End Time for intrusion in following format (HH:MM:SS): ")
    while True:
        _, frame = cap.read()
        #frame_id +=1
        height, width, channels = frame.shape

        #grayscale Image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[3] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 1.8)
                    y = int(center_y - h / 1.8)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        cv2.putText(frame, "DateTime: " + str(datetime.now().strftime("%d-%m-%Y-%H:%M:%S")), (10, 50), font, 2, (0, 0, 0), 3)
        cv2.putText(frame, "Press 'q' to quit",((10,430)), font, 2, (255,0,0), 2)
        time_now=datetime.now().strftime("%H:%M:%S")
        if time_now>=start_time and time_now<=end_time:
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    if label=="person":
                        #out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'),15.,(640,480))
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label , (x, y + 30), font, 2, color, 2)
                        #cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)
                        #image capturing
                        cv2.imwrite('images/frames.jpg', frame)
                        
                        
            
            #video making
            if len(faces) + len(bodies) > 0:
                #cv2.imwrite('images/frames.jpg', frame)
                if detecting_face:
                    timer_started=False
                else:
                    detecting_face=True
                    #current_time=datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                    #output_video=cv2.VideoWriter(f"videos/{current_time}.mp4",fourcc,20,frame_size)
                    print("Started Recording")
                    #sendsms()
                    #sendmail()
                    print("Mail Sent")
            elif detecting_face:
                if timer_started:
                    if time.time()-detection_stopped_time >= Seconds_to_record_after_detection:
                        detecting_face=False
                        timer_started=False
                        output_video.release()
                        print("Stopped Recording")
                else:
                    timer_started=True
                    detection_stopped_time=time.time()


            if detecting_face:
                output_video.write(frame)
        



        #elapsed_time = time.time() - starting_time
        #fps = frame_id / elapsed_time
        #cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
        cv2.imshow("Human Detection Frame", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    output_video.release()
    cap.release()
    cv2.destroyAllWindows()


detect_human()

