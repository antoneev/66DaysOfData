import requests

language_code = "en_US"
word_id = ["example","self-evaluating","observation"]
returnResults = {}

# results = [{"word":"example","phonetics":[{"text":"/ɪɡˈzæmpəl/","audio":"https://lex-audio.useremarkable.com/mp3/example_us_3.mp3"}],"meanings":[{"partOfSpeech":"verb","definitions":[{"definition":"Be illustrated or exemplified.","example":"the extent of Allied naval support is exampled by the navigational specialists provided"}]},{"partOfSpeech":"noun","definitions":[{"definition":"A thing characteristic of its kind or illustrating a general rule.","synonyms":["specimen","sample","exemplar","exemplification","instance","case","representative case","typical case","case in point","illustration"],"example":"it's a good example of how European action can produce results"},{"definition":"A person or thing regarded in terms of their fitness to be imitated or the likelihood of their being imitated.","synonyms":["precedent","lead","guide","model","pattern","blueprint","template","paradigm","exemplar","ideal","standard"],"example":"it is vitally important that parents should set an example"}]}]}]
# results = {"title":"No Definitions Found","message":"Sorry pal, we couldn't find definitions for the word you were looking for.","resolution":"You can try the search again at later time or head to the web instead."}

def requestDefinition(word_id):
    for word in word_id:
        url = "https://api.dictionaryapi.dev/api/v2/entries/" + language_code + "/" + word
        r = requests.get(url)
        results = r.json()

        parseJSON(results)

    print(returnResults)

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
        print("No Results Found")


returnResults = {'example': ['/ɪɡˈzæmpəl/', 'https://lex-audio.useremarkable.com/mp3/example_us_3.mp3', 'verb', 'Be illustrated or exemplified.', 'noun', 'A thing characteristic of its kind or illustrating a general rule.'], 'observation': ['/ˌɑbzərˈveɪʃ(ə)n/', 'https://lex-audio.useremarkable.com/mp3/observation_us_1.mp3', 'noun', 'The action or process of observing something or someone carefully or in order to gain information.']}
#print(returnResults)

for key, value in returnResults.items():
    print(key)
    for i in value:
        print(i)