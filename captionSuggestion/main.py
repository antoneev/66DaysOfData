import colorDetection
import objectDetection
import similarWordSuggestion
import cv2 as cv
import time

AllItems = []

if __name__ == "__main__":
    start = time.time()
    img = cv.imread("imgs/image-1.jpeg")
    colorDetection.numberOfClusters = 3
    colorDetection.main(img)
    objectDetection.main(img)

    if len(colorDetection.photoColors) > 0:
        AllItems.append(colorDetection.rootColors)
    else:
        print("No Colors Detected")

    if len(objectDetection.ListofObjects) > 0:
        AllItems.append(objectDetection.ListofObjects)
    else:
        print("No Objects Detected")

    similarWordSuggestion.maxReturnWords = 5
    similarWordSuggestion.main()

    print('---LIST OF ALL OBJECTS---')
    for i in range(len(AllItems)):
        for j in range(len(AllItems[i])):
            print(AllItems[i][j])

    end = time.time()
    print(end - start)