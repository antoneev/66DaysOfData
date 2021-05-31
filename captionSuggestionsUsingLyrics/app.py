import streamlit as st
from PIL import Image
import colorDetection
import lyricsGenius
import objectDetection
import main as backend
import os
import loadingFiles

# Upload image
import similarWordSuggestion

def downloadingExternalFile():
    st.info("Please wait the needed files are being retrieved. This will only happen on the initial load!")

    # Create needed folders
    path_images = "outputImgs/"
    path_lyrics = "outputLyrics/"

    os.makedirs(path_images)
    os.makedirs(path_lyrics)

    loadingFiles.main()

def displayOuput(artistSession):
    # Output Display
    st.write("# Output")

    # Output Images
    st.write("## Images")
    col1, col2 = st.beta_columns(2)

    # Output of objects
    col1.header("Object Detection")
    col1.write("### Objects Found")
    if len(objectDetection.ListofObjects) > 0:
        for i in objectDetection.ListofObjects:
            col1.write(i)
        objectDetectionImg = Image.open("outputImgs/objectDetected.png")
        col1.image(objectDetectionImg, use_column_width=True)
    else:
        col1.warning("No Objects Detected")

    # Output of colors
    col2.header("Color Detection")
    col2.write("### Colors Found")
    for i in colorDetection.rootColors:
        col2.write(i)
    colorDetectionImg = Image.open("outputImgs/colorDetected.png")
    col2.image(colorDetectionImg, use_column_width=True)

    # Show trigger similar words ONLY when objects are found
    st.header("# Similar Word Suggestion")
    if len(objectDetection.ListofObjects) > 0:
        st.write('### Similar words searched: ')
        similarWordFounds = backend.similarWordSuggestion.allSimilarWords

        for key, value in similarWordFounds.items():
            st.write('**Object:** ', key)
            for i in range(len(value)):
                st.write('**Similar Object: **', value[i])
    else:
        st.warning("Word Suggestion Algorithm Not Ran")

    # Outputting Lyrics
    st.header("Lyrics")

    if artistSession == 'Timed Out':
        st.error('Lyrics Genius Timed Out. Try again!')
    elif artistSession == 'Complete':
        st.write("### Songs Searched: ")
        for i in backend.lyricsGenius.allSongTitles:
            st.write(i)

        if len(backend.lyricsGenius.allLyrics) != 0:
            st.write('### All Lyrics Found: ')
            for i in backend.lyricsGenius.allLyrics:
                key = i.split('_')
                st.write('**Song:** ', key[0], '| **Element:** ', key[1], ' | **Lyrics:**',
                         backend.lyricsGenius.allLyrics[i])
    else:
        st.write("### Songs Searched: ")
        for i in backend.lyricsGenius.allSongTitles:
            st.write(i)
        st.warning("No Lyrics Found")

def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
    st.write("""
             # Caption Suggestion using Lyrics 🎶
             ## #66DaysOfData Project 1️⃣
             """)
    st.write(
        "GitHub Repo [Click Here!](https://github.com/antoneev/66DaysOfData/tree/main/captionSuggestionsUsingLyrics)")

    image_file = st.file_uploader("Upload Files",type=['png','jpeg','jpg'])
    if image_file is not None:
        img = load_image(image_file)
        st.image(img, width=250)

    # Color number of clusters
    numberOfColors = st.slider('How many colors would you like to search from?', 1, 10)

    # Input artist
    artistName = st.text_input("Please enter in the artist whose lyrics you'd like to search for.", 'Drake')

    # Input max songs for search
    numberOfSongs = st.slider('How many songs would you like to search from?', 1, 20)

    if st.button("Search") == True:

        if str(image_file) != 'None': # Verifying photo has been uploaded
            st.info('Algorithm is at work ...')

            # Saving file
            with open(os.path.join("outputImgs/", image_file.name), "wb") as f:
                f.write(image_file.getbuffer())

            file_path = "outputImgs/"+image_file.name

            # Converting numbers to pass to function
            numberOfColors = int(numberOfColors)
            numberOfSongs = int(numberOfSongs)

            # Calling algorithm
            artistSession, totalTime = backend.main(file_path, numberOfColors, artistName, numberOfSongs)

            # Calling display
            displayOuput(artistSession)

            # Outputting time
            st.success("Algorithm Completed in "+ str(round(totalTime, 2)) +" seconds")

            # Clear output button
            st.button('Clear Output')

            # Clearing all lists as streamlit caches this information
            backend.AllItems.clear()
            objectDetection.ListofObjects.clear()
            colorDetection.rootColors.clear()
            colorDetection.ListofColors.clear()
            colorDetection.photoColors.clear()
            lyricsGenius.allSongTitles.clear()
            lyricsGenius.artist_dict.clear()
            lyricsGenius.allLyrics.clear()
            similarWordSuggestion.allSimilarWords.clear()

        else:
            st.error("All fields must be filled out!")
    else:
        st.info('Please click the Search button after adding the needed information.')

if __name__ == '__main__':
    if not os.path.exists("files"):
        downloadingExternalFile()
    main()