import cv2 as cv

# Initializing variables
config_file = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'files/frozen_inference_graph.pb'
classFile = 'files/coco.names'

# Declare variables
ListofObjects = []
classLabels = []

def main(img):
    print('Object Detection started...') # Indicating algorithm started

    model = cv.dnn_DetectionModel(frozen_model,config_file) # Declaring model

    # Opening object labels
    with open(classFile,'rt') as f:
        classLabels = f.read().rstrip('\n').split('\n')

    # Model config
    model.setInputSize(320,320) # Image size based on config file
    model.setInputScale(1.0/127.5) # 255/2 = 127.5
    #moblienet => [-1,1]
    model.setInputMean((127.5,127.5,127.5))
    model.setInputSwapRB(True) # Automatically define images to be B&W
    ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.5) #Confidence level of objects to return

    font_scale = 1 # Font size
    font = cv.FONT_HERSHEY_PLAIN # Font type
    try:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            cv.rectangle(img,boxes,(255,0,0),2) # Places boxes around image
            # Outputs the text around image
            cv.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]-5), font, fontScale=font_scale, color=(0,255,0), thickness=1)
            # Checks if object is already in list
            if classLabels[ClassInd-1] not in ListofObjects and classLabels[ClassInd-1] != 'person':
                # Places new object in list
                ListofObjects.append(classLabels[ClassInd-1])
    except:
        return print('Object Detection exited...') # Exits algorithm if no object found

    #Outputs list of objects
    #print(ListofObjects)
    #Outputs image with boxes and text
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.show()

    cv.imwrite('outputImgs/objectDetected.png', img) # Saves algorithm to output folder

    #cv.imshow("Output", img)
    print('Object Detection completed...') # Indicating algorithm completed
    # cv2.waitKey(0)
    #main.ListOfAllElements.append(ListofObjects)