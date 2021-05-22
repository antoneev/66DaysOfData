import colorDetection
import objectDetection
import similarWordSuggestion
import cv2 as cv
import time
import lyricsGenius
import lyricsgenius as genius

AllItems = [] # Declare variable

def main(image_file,numberOfColors,artistName,numberOfSongs):
    similarWords = []
    start = time.time() # Starting time

    # Initializing variables
    img = cv.imread(image_file)
    colorDetection.numberOfClusters = numberOfColors
    similarWordSuggestion.maxReturnWords = 5
    api = genius.Genius('1M3VJ1T9KKthsOiN3BH1N9xD9BnOmKHK52b_vaCfJSQhUhRNTtDxgKjIxrvg0-DD')
    artist = artistName
    maxSongs = numberOfSongs
    sortBy = "title"

    colorDetection.main(img) # Calling color detection function
    objectDetection.main(img) # Calling object detection function

    # Either adding color/object to allitems list or displaing no colors/objects found
    if len(colorDetection.rootColors) > 0:
        AllItems.append(colorDetection.rootColors)
    else:
        print("No Colors Detected")

    if len(objectDetection.ListofObjects) > 0:
        AllItems.append(objectDetection.ListofObjects)
    else:
        print("No Objects Detected")

    # Adding similar words to a list then the list to overall list
    similarWordSuggestion.main()  # Calling similar word suggestion
    for key, value in similarWordSuggestion.allSimilarWords.items():
        for i in range(len(value)):
            similarWords.append(value[i])
    AllItems.append(similarWords)

    print('All colors and objects found: ' + str(AllItems)) # Displaying color and objects

    # Passing each element to the lyricsGenius function 1 by 1
    for i in range(len(AllItems)):
        # Eliminates creating the same JSON file n times
        if i == 0:
            artistFile = lyricsGenius.findArtistSongs(api, artist, maxSongs, sortBy)
            if artistFile == 'Timeout: Request timed out: HTTPSConnectionPool':
                break
        for j in range(len(AllItems[i])):
            currentElement = AllItems[i][j]
            print('\nSearching for element:', currentElement)
            lyricsGenius.main(currentElement)

    # Displays if any lyrics were found or not
    if artistFile == 'Timeout: Request timed out: HTTPSConnectionPool':
        print('Lyrics Genius Timed Out')
        artistSession = 'Timed Out'
    elif len(lyricsGenius.allLyrics) > 0:
        print(lyricsGenius.allLyrics)
        artistSession = 'Complete'
    else:
        print("No Lyrics Found. Please try another image or artist.")
        artistSession = 'No Lyrics Found'

    # Outputting summary of all elements. If no elements found empty arrays are shown.
    print('\n-----Summary START------')
    print('Colors Found: ', colorDetection.rootColors)
    print('Objects Found: ', objectDetection.ListofObjects)
    print('Similar Words Found: ',similarWordSuggestion.allSimilarWords)
    print('All Colors and Objects Found: ', str(AllItems))
    print('Songs Searched: ', lyricsGenius.allSongTitles)
    print('All Lyrics Found: ', lyricsGenius.allLyrics)
    print('-----Summary END--------\n')

    end = time.time() # Calculates end time
    totalTime = (end - start)
    print(totalTime) # Prints times in seconds

    return artistSession, totalTime

if __name__ == "__main__":
    main()