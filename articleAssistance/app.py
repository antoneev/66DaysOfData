import base64
import streamlit as st
import os
import main as backend
import wget
import shutil
import handleDatabase
import webbrowser as wb

# Creating side bar with options
menu = ["Home","Upload Article","Upload Plain Text"]
choice = st.sidebar.selectbox("Menu",menu)

def createFiles():
    st.info("Please wait the needed files are getting created. This will only happen on the initial load!")

    # Create needed folders
    inputFolder = "input/"
    outputFolder = "output/"
    dbFolder = "db/"

    os.makedirs(inputFolder)
    os.makedirs(outputFolder)
    os.makedirs(dbFolder)

    # Downloading needed files
    urlHomeExample = "https://dl.dropbox.com/s/c2fzht782nfzcr5/home-example.pdf" # change www to dl and remove dl from behind zip to download
    filename = wget.download(urlHomeExample)
    shutil.move(filename, "output/")

    urlDB = "https://dl.dropbox.com/s/apczyhy91f0hiw3/database.sql" # change www to dl and remove dl from behind zip to download
    filename = wget.download(urlDB)
    shutil.move(filename, "db/")

    # Creating and loading DB table
    handleDatabase.createTable()

# Displaying PDF files on UI
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    # On page load or clicking the Home selection
    if choice == "Home":
        # Writing title and subtitle
        st.write("""
                     # Article Assistance
                     ## #66DaysOfData Project ✌🏾
                     """)
        # Writing GitHub Link
        st.write(
            "GitHub Repo [Click Here!](https://github.com/antoneev/66DaysOfData/tree/main/articleAssistance)")

        # Calling displayPDF to render the PDF to the screen
        displayPDF('output/home-example.pdf')

    # Clicking the Upload Article selection
    if choice == "Upload Article":
        # File Upload
        image_file = st.file_uploader("Upload Files", type=['pdf'])
        # Verifying button is clicked to start algorithm
        if st.button("Start Algorithm") == True:
            # Checking if a file was uploaded
            if str(image_file) != 'None':
                st.info('Algorithm is at work ...')

                # Saving file
                with open(os.path.join("input/", image_file.name), "wb") as f:
                    f.write(image_file.getbuffer())

                filePath = "input/" + image_file.name
                inputType = 'PDF'

                # Putting block of code in a try catch in case the API fails
                try:
                    # Calling the backend and passing the file
                    outputFilePath = backend.main(inputType, filePath)
                    # Rendering the file to the UI
                    displayPDF(outputFilePath)
                    # Success message
                    st.success('Algorithm completed successfully!')
                except:
                    # Exception message
                    st.error("☹️ Something went wrong. Please try again later!")

            else:
                # File validation message
                st.error('Please upload a file.')
    # Clicking Uploading Plain Text selection
    if choice == "Upload Plain Text":
        # Adding textarea
        text = st.text_area("Copy and paste text from article below:")
        # Verifying button is clicked to start algorithm
        if st.button("Start Algorithm") == True:
            # Verifying the textbox is not empty
            if text == "":
                st.warning("Please enter text into the above textarea.")
            else:
                st.info('Algorithm is at work ...')
                inputType = 'PlainText'

                # Putting block of code in a try catch in case the API fails
                try:
                    # Calling the backend and passing the file
                    outputFilePath = backend.main(inputType, text)
                    # Rendering the file to the UI
                    displayPDF(outputFilePath)
                    # Success message
                    st.success('Algorithm completed successfully!')
                except:
                    # Exception message
                    st.error("☹️ Something went wrong. Please try again later!")

if __name__ == '__main__':
    # Creating needed folders on load
    if os.path.isdir("input/") == False:
        createFiles()
    main()
