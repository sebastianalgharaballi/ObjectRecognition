import cv2
from buttons import Buttons

# Initialize buttons
button = Buttons()
button.add_button('person', 20, 20)
button.add_button('cell phone', 20, 100)
button.add_button('backpack', 20, 180)
button.add_button('umbrella', 20, 260)
button.add_button('frisbee', 20, 340)
button.add_button('sports ball', 20, 420)
button.add_button('fork', 20, 500)
button.add_button('remote', 20, 580)

# DNN for openCV
net = cv2.dnn.readNet('dnn_model/yolov4-tiny.weights', 'dnn_model/yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (320, 320), scale = 1/255)

# Load object lists
classes = []
with open('dnn_model/classes.txt', 'r') as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print('Objects')
print(classes)


# Camera initialization for 720p Macbook Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN: # left click on mouse to see corresponding (x, y) coordinates on camera
        button.button_click(x, y)


# Create window
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_button)

while True:
    # Generate Frames
    ret, frame = cap.read()

    # Generate active buttons list
    active_buttons = button.active_buttons_list()
    print('Active Buttons', active_buttons)

    # Detect Objects
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2) # text above item
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3) # code for rectangular detection

    # Show buttons
    button.display_buttons(frame)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)

cap.release() # ensure camera can be used by other applications while program is in use
