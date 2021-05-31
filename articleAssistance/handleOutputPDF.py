from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate

doc = SimpleDocTemplate("outputFile/articleAssistance.pdf")

def outputFile(article,complexWords,returnResults):
    info = []
    partsOfSpeech = ['noun', 'pronoun', 'verb', 'adjective', 'adverb', 'preposition', 'conjunction', 'interjection']

    complexWordsInfo = "<font size = '12'> "
    for key, value in returnResults.items():
        count = 0
        complexWordsInfo += "<font color ='red'>" + key + " </font>"
        for i in value:
            if i in partsOfSpeech:
                complexWordsInfo += "<br></br><strong> " + i + "</strong> "
            else:
                complexWordsInfo += i + " "
            count += 1
        complexWordsInfo += "<br></br><br></br>"
    complexWordsInfo += "</font>"

    tokenizeList = [sub.split() for sub in article]

    paraContent = "<font size = '12'> "
    for i in range(len(tokenizeList)):
        for j in range(len(tokenizeList[i])):
            if tokenizeList[i][j].lower() in complexWords:
                paraContent += "<font color ='red'>" + tokenizeList[i][j] + "</font color> "
            else:
                paraContent += tokenizeList[i][j] + " "
    paraContent += "</font>"

    lineBreak = "<br></br><br></br>"
    title = "<font size = '15'><strong> Article Assistance </strong></font>"
    subtitle = "<font size = '15'><strong> #66DaysOfData Project 2 </strong></font>"

    info.append(Paragraph(title))
    info.append(Paragraph(subtitle))
    info.append(Paragraph(lineBreak))
    info.append(Paragraph(complexWordsInfo))
    info.append(Paragraph(lineBreak))
    info.append(Paragraph(paraContent))

    doc.build(info)