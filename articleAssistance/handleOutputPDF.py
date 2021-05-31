from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate

# Creating outfile object
outputFilePath = "output/articleAssistance.pdf"
doc = SimpleDocTemplate(outputFilePath)

def outputFile(article,complexWords,returnResults):
    # Creating list to add HTML components
    info = []
    # List of parts of speech
    partsOfSpeech = ['noun', 'pronoun', 'verb', 'adjective', 'adverb', 'preposition', 'conjunction', 'interjection']

    # Font size for complex word bounding box
    complexWordsInfo = "<font size = '12'> "
    # Looping through the dict
    for key, value in returnResults.items():
        # Setting all complex word to red
        complexWordsInfo += "<font color ='red'>" + key + " </font>"
        for i in value:
            # Checking if the word is a part of speech
            if i in partsOfSpeech:
                # Setting part of speech to bold
                complexWordsInfo += "<br></br><strong> " + i + "</strong> "
            else:
                # Attaching all other words to the paragraph normally
                complexWordsInfo += i + " "
            # Adding 2 breaklines between each definition
        complexWordsInfo += "<br></br><br></br>"
    complexWordsInfo += "</font>"

    # Tokenizing each article
    tokenizeList = [sub.split() for sub in article]

    # Font size for article bounding box
    paraContent = "<font size = '12'> "
    # Looping through list
    for i in range(len(tokenizeList)):
        # Looping through each word
        for j in range(len(tokenizeList[i])):
            # Checking if the word is in the complexWords list
            if tokenizeList[i][j].lower() in complexWords:
                # Setting each complex word to red
                paraContent += "<font color ='red'>" + tokenizeList[i][j] + "</font color> "
            else:
                # Attaching all other words to the paragraph normally
                paraContent += tokenizeList[i][j] + " "
    paraContent += "</font>"

    # Defining linebreak, title and subtitle
    lineBreak = "<br></br><br></br>"
    title = "<font size = '15'><strong> Article Assistance </strong></font>"
    subtitle = "<font size = '15'><strong> #66DaysOfData Project 2 </strong></font>"

    # Appending each to the info list
    info.append(Paragraph(title))
    info.append(Paragraph(subtitle))
    info.append(Paragraph(lineBreak))
    info.append(Paragraph(complexWordsInfo))
    info.append(Paragraph(lineBreak))
    info.append(Paragraph(paraContent))

    # Using the info and SimpleDocTemplate to render the PDF
    doc.build(info)

    return outputFilePath