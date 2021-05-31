import handleText
import handleDatabase
import handleRetrievingWords
import handleOutputPDF

# Handling words to identify those considered complex
def unknown(cleanedWords):
    # Creating list for words
    unknownWords = []

    # Checking each word
    for word in cleanedWords:
        # Calling the DB to see if word is within or not
        result = handleDatabase.searchForWords(word)
        # If the word is not in the DB
        if result == False:
            # Checking the syllable count of the word
            count = handleText.syllable_count(word)
            # If the count if greater than 2 including it in the unknownWords list
            if count > 2:
                unknownWords.append(word)

    # Returning list of words identified as complex
    return unknownWords

def main(check,userInput):
    # Checking if function was called from PDF selection on UI
    if check == 'PDF':
        # Passes in user PDF file while calling pdfToText
        article = handleText.pdfToText(userInput)
        # Passes in article while calling wordCleaningPDF
        cleanedWords = handleText.wordCleaningPDF(article)
        # Passes in cleaned data while calling unknown
        searchForWords = unknown(cleanedWords)
    else:
        # Creating article list as the wordCleaningText handles lists to tokenize data correctly
        article = []
        # Appends user input from UI
        article.append(userInput)
        # Passes in article while calling wordCleaningText
        cleanedWords = handleText.wordCleaningText(article)
        # Passes in cleaned words while calling unknown
        searchForWords = unknown(cleanedWords)

    # Passes words identified as complex to API
    definitionsRetrieved = handleRetrievingWords.requestDefinition(searchForWords)

    print('------- SUMMARY START --------')
    print('Article:', article)
    print('Complex words:', searchForWords)
    print('Google: ', definitionsRetrieved)
    print('------- SUMMARY END --------')

    # Creating PDF output File
    outputFilePath = handleOutputPDF.outputFile(article,searchForWords,definitionsRetrieved)

    return outputFilePath

if __name__ == '__main__':
    main()