import cv2 as cv

#List of objects
ListofObjects = []

def main(img):
    print('Object Detection started...')
    config_file = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'files/frozen_inference_graph.pb'

    model = cv.dnn_DetectionModel(frozen_model,config_file)

    classLabels = []
    classFile = 'files/coco.names'
    with open(classFile,'rt') as f:
        classLabels = f.read().rstrip('\n').split('\n')

    #print(classLabels)
    #print(len(classLabels))

    ##Model Config
    #image size based on config file
    model.setInputSize(320,320)
    #255/2 = 127.5
    model.setInputScale(1.0/127.5)
    #moblienet => [-1,1]
    model.setInputMean((127.5,127.5,127.5))
    # automatically define images to be B&W
    model.setInputSwapRB(True)

    #read an image

    #Confidence level of objects to return
    ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.5)

    #print(ClassIndex)

    #Font size and font for output
    font_scale = 1
    font = cv.FONT_HERSHEY_PLAIN
    try:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            #places boxes around image
            cv.rectangle(img,boxes,(255,0,0),2)
            #outputs the text around image
            cv.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]-5), font, fontScale=font_scale, color=(0,255,0), thickness=1)
            #checks if object is already in list
            if classLabels[ClassInd-1] not in ListofObjects and classLabels[ClassInd-1] != 'person':
                #places new object in list
                ListofObjects.append(classLabels[ClassInd-1])
    except:
        return print('Object Detection complete...')

    #Outputs list of objects
    #print(ListofObjects)

    #Outputs image with boxes and text
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.show()
    cv.imwrite('outputImgs/objectDetected.png', img)

    cv.imshow("Output", img)
    print('Object Detection complete...')
    # cv2.waitKey(0)
    #main.ListOfAllElements.append(ListofObjects)