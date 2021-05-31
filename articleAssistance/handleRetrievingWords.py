import requests

# Setting language code and creating dict for results
language_code = "en_US"
returnResults = {}

# Requesting Definition from API
def requestDefinition(word_id):
    # Looping through each word in complex word list
    for word in word_id:
        # Defining URL
        url = "https://api.dictionaryapi.dev/api/v2/entries/" + language_code + "/" + word
        # Defining request
        r = requests.get(url)
        # Return JSON from results
        results = r.json()

        # Calling Parse JSON to find needed results from resulted JSON
        parseJSON(results)

    return returnResults

def parseJSON(results):
    #Defining list to save contents
    contents = []

    # Placing in a try/expect as some words may not return needed results as no definition is found
    try:
        # Looping through JSON
        for word in results:
            # Returning Audio file
            #contents.append(word['phonetics'][0]['text']) # Commented out as it does not display as expected on the PDF
            contents.append(word['phonetics'][0]['audio'])
            for defs in word['meanings']:
                # Returning part of speech
                contents.append(defs['partOfSpeech'])
                # Returning definition
                contents.append(defs['definitions'][0]['definition'])
        # Assigning the word as the dict key and all results to it
        returnResults[word['word']] = contents
    except:
        # Skipping words with no definitions found
        pass