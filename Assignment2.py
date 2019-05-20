import ai2thor.controller
import cv2
import argparse
import numpy as np
import keyboard

controller = ai2thor.controller.Controller()
controller.start(player_screen_width=640, player_screen_height=480)

is_visible = False
rotate_count = 0 
look_count = 0
event = controller.reset('FloorPlan28',)
event = controller.step(dict(action='Initialize', gridSize=0.25))
event = controller.step(dict(action='RotateRight'))
event = controller.step(dict(action='RotateRight'))
event = controller.step(dict(action = "MoveAhead"))
event = controller.step(dict(action = "MoveAhead"))
event = controller.step(dict(action = "MoveAhead"))
event = controller.step(dict(action = "MoveAhead"))
event = controller.step(dict(action = "MoveRight"))
event = controller.step(dict(action = "MoveRight"))
event = controller.step(dict(action = "MoveRight"))




# read class names from text file
classes = None
with open("yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")


# function to get the output layer names in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    
    # generate different colors for different classes 
    

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 3)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_frames(controller):
    last_event = controller.last_event
    rot = last_event.metadata['agent']['rotation']
    rot_y = rot['y']
    if rot_y == 360 or rot_y == -360:
        rot_y = 0
    if keyboard.is_pressed('w'):
        event = controller.step(dict(action='MoveAhead'))
    elif keyboard.is_pressed('s'):
        event = controller.step(dict(action='MoveBack'))
    elif keyboard.is_pressed('a'):
        event = controller.step(dict(action='MoveLeft'))
    elif keyboard.is_pressed('d'):
        event = controller.step(dict(action='MoveRight'))
    elif keyboard.is_pressed('up'):
        event = controller.step(dict(action='LookUp'))
    elif keyboard.is_pressed('down'):
        event = controller.step(dict(action='LookDown'))
    elif keyboard.is_pressed('left'):
        rot['y'] = rot_y - 10
        event = controller.step(dict(action='Rotate', rotation=rot))
    elif keyboard.is_pressed('right'):
        rot['y'] = rot_y + 10
        event = controller.step(dict(action='Rotate', rotation=rot))
    elif keyboard.is_pressed('f'):
        objects = [o for o in last_event.metadata['objects'] if o['visible'] and o['openable']]
        if len(objects) == 0:
            event = last_event
        objects.sort(key=lambda o: o['distance'])
        nearest_obj = objects[0]
        if nearest_obj['isopen']:
            event = controller.step(dict(action='CloseObject', objectId=nearest_obj['objectId']))
        else:
            event = controller.step(dict(action='OpenObject', objectId=nearest_obj['objectId']))
    elif keyboard.is_pressed('p'):
        cv2.destroyAllWindow()
    else:
        event = last_event

    return event.cv2img
    
conf_threshold = 0.5
nms_threshold = 0.4

while True:
    image = get_frames(controller)

    Width = image.shape[1]
    Height = image.shape[0]
    dim = (Width,Height)
    image = cv2.resize(image,dim,interpolation = cv2.INTER_AREA)
    scale = 0.00392
    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    # run inference through the network and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []

    # for each detetion from each output layer: get the confidence, class id, bounding box params and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    # display output image    
    cv2.imshow("object detection", image)

    # pauses for 0.001 seconds before fetching next image
    key = cv2.waitKey(10) 
    
    #if ESC is pressed, exit loop
    if key == 27:
        cv2.destroyAllWindows()
        break