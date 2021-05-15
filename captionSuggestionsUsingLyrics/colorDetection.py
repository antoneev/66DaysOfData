import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('files/colors.csv', names=index, header=None)
ListofColors = []
photoColors = []
rootColors = []
numberOfClusters = 0

def compare_img(img_1, img_2):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off')
    ax[1].axis('off')
    f.tight_layout()
    #plt.show()
    f.savefig('outputImgs/colorDetected.png')

def palette(clusters):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    steps = width/clusters.cluster_centers_.shape[0]
    for idx, centers in enumerate(clusters.cluster_centers_):
        palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
    return palette

def all_colors_in_img(color_codes):
    #Coverts the 3d array of all colors into a 1d list
    for ij in np.ndindex(color_codes.shape[:2]):
        #print(ij, palette(clt_1)[ij])
        ListofColors.append(color_codes[ij])
    return ListofColors

#Color Recognition using Colors CSV
def recognize_color(color_index):
    R = color_index[0]
    G = color_index[1]
    B = color_index[2]
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

def predict_color(unique_colors):
    for i in range(len(unique_colors)):
        colorFound = recognize_color(unique_colors[i])
        photoColors.append(colorFound)
    return photoColors

def return_root_color(allColorsFound):
    for i in range(len(allColorsFound)):
        if allColorsFound not in rootColors:
            rootColors.append(allColorsFound[i].split()[-1].lower())

def main(img):
    print('Color Detection started...')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # resize image
    dim = (500, 300)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    # define KNN
    clt = KMeans(n_clusters=numberOfClusters)
    clt.fit(img.reshape(-1, 3))
    clt.labels_
    clt.cluster_centers_

    # display color palette
    clt_1 = clt.fit(img.reshape(-1, 3))
    compare_img(img, palette(clt_1))
    # print(palette(clt_1))

    ListofColors = all_colors_in_img(palette(clt_1))

    # Finds the unique arrays in the list
    unique_rows = np.unique(ListofColors, axis=0)
    #print(unique_rows)
    photoColors = predict_color(unique_rows)
    return_root_color(photoColors)
    #print(photoColors)
    print('Color Detection complete...')


if __name__ == '__main__':
    main()