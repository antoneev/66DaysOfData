import slate3k as slate
import re

# Extracting words from PDF file
def pdfToText(filePath):
    with open(filePath,'rb') as f:
        extracted_text = slate.PDF(f)
    return extracted_text

def wordCleaningPDF(textFromPDF):
    # Split new lines returned from reading file
    dataNoNewLines = []

    # Removing new lines
    pdf_miner = [x for x in textFromPDF[0].split("\n") if x != ""]

    # Removing hex '\x' data
    for i in range(len(pdf_miner)):
        dataNoNewLines.append(pdf_miner[i].lstrip())

    print('dataNoNewLines', dataNoNewLines)

    # Call the wordCleaningText function
    cleanedList = wordCleaningText(dataNoNewLines)

    # Returning cleaned data
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

    # Returning clean data
    return cleanedList

def syllable_count(word):
    # Setting count to 0 and defining all vowels
    count = 0
    vowels = "aeiouy"

    # Checking if first letter of the word is a vowel
    if word[0] in vowels:
        # Increase count
        count += 1

    # Checking if all other words other than the first letter are vowels
    for index in range(1, len(word)):
        # if the word doesn't have two vowels back to back increase the count by 1
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1

    # Decreasing count if word ends in e
    if word.endswith("e"):
        count -= 1

    # Increasing count to 0 if it didn't meet any of the above requirements
    if count == 0:
        count += 1
    return count