import requests

language_code = "en_US"
returnResults = {}

def requestDefinition(word_id):
    for word in word_id:
        url = "https://api.dictionaryapi.dev/api/v2/entries/" + language_code + "/" + word
        r = requests.get(url)
        results = r.json()

        parseJSON(results)

    return returnResults

def parseJSON(results):
    contents = []
    try:
        for word in results:
            #contents.append(word['phonetics'][0]['text'])
            contents.append(word['phonetics'][0]['audio'])
            for defs in word['meanings']:
                contents.append(defs['partOfSpeech'])
                contents.append(defs['definitions'][0]['definition'])
        returnResults[word['word']] = contents
    except:
        pass