import handleText
import handleDatabase
import handleRetrievingWords
import handleOutputPDF

# TODO
# Can screenshots words be extracted?
# UI - Home Page

def unknown(cleanedWords):
    unknownWords = []

    for word in cleanedWords:
        result = handleDatabase.searchForWords(word)
        if result == False:
            count = handleText.syllable_count(word)
            if count > 2:
                unknownWords.append(word)

    return unknownWords

def main(check,userInput):
    if check == 'PDF':
        article = handleText.pdfToText(userInput)
        cleanedWords = handleText.wordCleaningPDF(article)
        searchForWords = unknown(cleanedWords)
    else:
        article = []
        article.append(userInput)
        cleanedWords = handleText.wordCleaningText(article)
        searchForWords = unknown(cleanedWords)

    definitionsRetrieved = handleRetrievingWords.requestDefinition(searchForWords)
    print('Article:', article)
    print('Complex words:', searchForWords)
    print('Google: ', definitionsRetrieved)
    handleOutputPDF.outputFile(article,searchForWords,definitionsRetrieved)

if __name__ == '__main__':
    main()