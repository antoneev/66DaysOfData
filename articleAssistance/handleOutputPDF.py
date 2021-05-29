from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate
from PIL import Image
import webbrowser

doc = SimpleDocTemplate("outputFile/articleAssistance.pdf")

text = ["It’s day 1, you’re being shown off to senior execs and team members. Orientation is now over, and your supervisor for the next 12 weeks asks, “What do you want to learn?"]
words = ['Orientation', 'team']
returnResults = {'example': ['https://lex-audio.useremarkable.com/mp3/example_us_3.mp3', 'verb', 'Be illustrated or exemplified.', 'noun', 'A thing characteristic of its kind or illustrating a general rule.'], 'observation': ['https://lex-audio.useremarkable.com/mp3/observation_us_1.mp3', 'noun', 'The action or process of observing something or someone carefully or in order to gain information.']}

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


tokenizeList = [sub.split() for sub in text]

paraContent = "<font size = '12'> "
for i in range(len(tokenizeList)):
    for j in range(len(tokenizeList[i])):
        if tokenizeList[i][j] in words:
            paraContent += "<font color ='red'>" + tokenizeList[i][j] + "</font color> "
        else:
            paraContent += tokenizeList[i][j] + " "
paraContent += "</font>"

lineBreak = "<br></br><br></br>"
title = "<font size = '15'><strong> Article Assistance </strong></font>"
subtitle = "<font size = '15'><strong> #66DaysOfData Project 2 </strong></font>"

info = []

info.append(Paragraph(title))
info.append(Paragraph(subtitle))
info.append(Paragraph(lineBreak))

info.append(Paragraph(complexWordsInfo))

info.append(Paragraph(lineBreak))
info.append(Paragraph(paraContent))

doc.build(info)

#webbrowser.open_new('articleAssistance-1.pdf')