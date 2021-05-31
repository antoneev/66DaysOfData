import slate3k as slate
import re

def pdfToText(filePath):
    with open(filePath,'rb') as f:
        extracted_text = slate.PDF(f)
    return extracted_text

def wordCleaningPDF(textFromPDF):
    # Split new lines returned from reading file
    dataNoNewLines = []
    pdf_miner = [x for x in textFromPDF[0].split("\n") if x != ""]
    for i in range(len(pdf_miner)):
        dataNoNewLines.append(pdf_miner[i].lstrip())

    cleanedList = wordCleaningText(dataNoNewLines)
    return cleanedList

def wordCleaningText(untokenizeText):
    # Tokenize each sentence
    tokenizeList = [sub.split() for sub in untokenizeText]

    # Remove all symbols other than apostrophe and remove numeric numbers
    cleanedList = []
    for i in range(len(tokenizeList)):
        for j in range(len(tokenizeList[i])):
            cleanedWord = re.sub(r"[^a-zA-Z’-]", '', tokenizeList[i][j]).lower()
            if cleanedWord not in cleanedList:
                cleanedList.append(cleanedWord)

    # Remove all empty strings from list
    while ("" in cleanedList):
        cleanedList.remove("")

    return cleanedList

def syllable_count(word):
    count = 0
    vowels = "aeiouy"

    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count