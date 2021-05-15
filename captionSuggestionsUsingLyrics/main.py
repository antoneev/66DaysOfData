import colorDetection
import objectDetection
import similarWordSuggestion
import cv2 as cv
import time
import lyricsGenius
import lyricsgenius as genius

AllItems = []
allSongs = {}

#TODO
#Work on rejecting anything other than number/letters or a space in the artist name
#Handle if the color name ends in () get the second to last name
#Handling similar words 

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

    api = genius.Genius('1M3VJ1T9KKthsOiN3BH1N9xD9BnOmKHK52b_vaCfJSQhUhRNTtDxgKjIxrvg0-DD')
    artist = "Andy Shauf"
    maxSongs = 3
    sortBy = "title"

    for i in range(len(AllItems)):
        for j in range(len(AllItems[i])):
            currentObject = AllItems[i][j]
            lyricsFound = lyricsGenius.main(api, artist, maxSongs, sortBy, currentObject)

    if len(lyricsGenius.allLyrics) > 0:
        print(lyricsGenius.allLyrics)
    else:
        print("No Lyrics Found. Please try another image or artist.")

    end = time.time()
    print(end - start)