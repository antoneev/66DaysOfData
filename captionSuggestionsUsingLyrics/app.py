import streamlit as st
from PIL import Image
import main as backend
import os

# TODO
# Handle car returning card?
# Handle word suggestion
# Handle reset

# Upload image
def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
    st.write("""
             # Caption Suggestion using Lyrics 🎶
             ## #66DaysOfData Project 1️⃣
             """)

    image_file = st.file_uploader("Upload Files",type=['png','jpeg','jpg'])
    if image_file is not None:
        img = load_image(image_file)
        st.image(img, width=250)

    # Color number of clusters
    numberOfColors = st.slider('How many colors would you like to search from?', 1, 10)

    # Trigger similar word search
    st.write("Would you like to search for similar words to the objects found?")

    # Input artist
    artistName = st.text_input("Please enter in the artist whose lyrics you'd like to search for.", 'Drake')

    # Input max songs for search
    numberOfSongs = st.slider('How many songs would you like to search from?', 1, 20)

    if st.button("Search"):
        st.info('Algorithm is at work ...')
        # Saving file
        with open(os.path.join("outputImgs/", image_file.name), "wb") as f:
            f.write(image_file.getbuffer())
        file_path = "outputImgs/"+image_file.name

        # Converting numbers to pass to function
        numberOfColors = int(numberOfColors)
        numberOfSongs = int(numberOfSongs)

        # Calling algorithm
        totalTime = backend.main(file_path, numberOfColors, artistName, numberOfSongs)

        # Output Display
        st.write("# Output")

        # Output Images
        st.write("## Images")
        col1, col2 = st.beta_columns(2)

        objectDetectionImg = Image.open("outputImgs/objectDetected.png")
        col1.header("Object Detection")
        col1.image(objectDetectionImg, use_column_width=True)

        colorDetectionImg = Image.open("outputImgs/colorDetected.png")
        col2.header("Color Detection")
        col2.image(colorDetectionImg, use_column_width=True)

        st.write("## Lyrics")

        #Outputting Lyrics
        st.write('All Lyrics Found')
        for i in backend.lyricsGenius.allLyrics:
            key = i.split('_')
            st.write('**Song:** ', key[0], '| **Element:** ', key[1], ' | **Lyrics:**', backend.lyricsGenius.allLyrics[i])

        st.success("Algorithm Completed in "+ str(round(totalTime, 2)) +" seconds")
    else:
        st.info('Please click Search button after adding the needed information')

if __name__ == '__main__':
    main()