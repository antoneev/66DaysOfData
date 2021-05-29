import handleText
import handleDatabase

def unknown(cleanedWords):
    unknownWords = []

    for word in cleanedWords:
        result = handleDatabase.searchForWords(word)
        if result == False:
            count = handleText.syllable_count(word)
            if count > 2:
                unknownWords.append(word)

    return unknownWords

def main(check):
    if check == 'PDF':
        textFromPDF = handleText.pdfToText()
        cleanedWords = handleText.wordCleaningPDF(textFromPDF)
        searchForWords = unknown(cleanedWords)
        print(searchForWords)
    else:
        text = ["It’s day 1, you’re being shown off to senior execs and team members. Orientationb is now over, and your supervisor for the next 12 weeks asks, “What do you want to learn?” What do you say? You can give the politically correct answer, possibly the answer you said in the interview, or you can say what you want to learn, but do you? Do you know what you want to do or are you hoping to be handed a 12-week syllabus? Often, I’ve learned the 12-week syllabus isn’t known as you’re told we’ve been placed on teams based on our skills. And most times your supervisor is just self-evaluating you the first few weeks to see what you can be useful for. But if you have a clear direction or some sort of general idea a good supervisor will be help you accomplish those goals over the 12-week span."]
        cleanedWords = handleText.wordCleaningText(text)
        searchForWords = unknown(cleanedWords)
        print(searchForWords)

if __name__ == '__main__':
    main("PDFs")