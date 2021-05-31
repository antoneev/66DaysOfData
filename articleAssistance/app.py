import base64
import streamlit as st
import os
import main as backend

menu = ["Home","Upload Article","Upload Plain Text"]
choice = st.sidebar.selectbox("Menu",menu)

def createFiles():
    st.info("Please wait the needed files are getting created. This will only happen on the initial load!")

    # Create needed folders
    path_images = "inputFile/"
    path_lyrics = "outputFile/"

    os.makedirs(path_images)
    os.makedirs(path_lyrics)

def displayPDF():
    with open('outputFile/articleAssistance.pdf', "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    st.markdown(pdf_display, unsafe_allow_html=True)

    st.success('Algorithm completed successfully!')

def main():
    if choice == "Home":
        st.write("""
                     # Article Assistance
                     ## #66DaysOfData Project ✌🏾
                     """)
        st.write(
            "GitHub Repo [Click Here!](https://github.com/antoneev/66DaysOfData/tree/main/articleAssistance)")

    if choice == "Upload Article":
        image_file = st.file_uploader("Upload Files", type=['pdf'])
        if st.button("Start Algorithm") == True:
            if str(image_file) != 'None':  # Verifying photo has been uploaded
                st.info('Algorithm is at work ...')

                # Saving file
                with open(os.path.join("inputFile/", image_file.name), "wb") as f:
                    f.write(image_file.getbuffer())

                filePath = "inputFile/" + image_file.name
                inputType = 'PDF'

                backend.main(inputType, filePath)

                displayPDF()

            else:
                st.error('Please upload a file.')

    if choice == "Upload Plain Text":
        text = st.text_area("Copy and paste text from article below:")
        if st.button("Start Algorithm") == True:
            if text == "":
                st.warning("Please enter text into the above textarea.")
            else:
                st.info('Algorithm is at work ...')
                inputType = 'PlainText'

                backend.main(inputType, text)
                displayPDF()

if __name__ == '__main__':
    if os.path.isdir("inputFile/") == False:
        createFiles()
    main()