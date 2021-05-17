import colorDetection
import objectDetection
import similarWordSuggestion
import cv2 as cv
import time
import lyricsGenius
import lyricsgenius as genius

AllItems = [] # Declare variable

if __name__ == "__main__":
    start = time.time() # Starting time

    # Initializing variables
    img = cv.imread("imgs/image-1.jpeg")
    colorDetection.numberOfClusters = 3
    similarWordSuggestion.maxReturnWords = 5
    api = genius.Genius('1M3VJ1T9KKthsOiN3BH1N9xD9BnOmKHK52b_vaCfJSQhUhRNTtDxgKjIxrvg0-DD')
    artist = "Andy Shauf"
    maxSongs = 3
    sortBy = "title"

    colorDetection.main(img) # Calling color detection function
    objectDetection.main(img) # Calling object detection function

    # Either adding color/object to allitems list or displaing no colors/objects found
    if len(colorDetection.photoColors) > 0:
        AllItems.append(colorDetection.rootColors)
    else:
        print("No Colors Detected")

    if len(objectDetection.ListofObjects) > 0:
        AllItems.append(objectDetection.ListofObjects)
    else:
        print("No Objects Detected")

    similarWordSuggestion.main() # Calling similar word suggestion
    print(similarWordSuggestion.allSimilarWords)

    print('All colors and objects found: ' + str(AllItems)) # Displaying color and objects

    # Passing each element to the lyricsGenius function 1 by 1
    for i in range(len(AllItems)):
        # Eliminates creating the same JSON file n times
        if i == 0:
            artistFile = lyricsGenius.findArtistSongs(api, artist, maxSongs, sortBy)
        for j in range(len(AllItems[i])):
            currentElement = AllItems[i][j]
            print('\nSearching for element:', currentElement)
            lyricsFound = lyricsGenius.main(artistFile, maxSongs, currentElement)

    # Displays if any lyrics were found or not
    if len(lyricsGenius.allLyrics) > 0:
        print(lyricsGenius.allLyrics)
    else:
        print("No Lyrics Found. Please try another image or artist.")

    # Outputting summary of all elements. If no elements found empty arrays are shown.
    print('\n-----Summary START------')
    print('Colors Found: ', colorDetection.photoColors)
    print('Objects Found: ', objectDetection.ListofObjects)
    print('Similar Words Found: ',similarWordSuggestion.allSimilarWords)
    print('All Colors and Objects Found: ', str(AllItems))
    print('All Lyrics Found: ', lyricsGenius.allLyrics)
    print('-----Summary END--------\n')

    end = time.time() # Calculates end time
    print(end - start) # Prints times in seconds