import json
import csv
import os

# Create an empty dictionary to store your songs and related data
artist_dict = {}
lyricsFound = {}
allLyrics = {}

def collectSongData(song):
    songInfo = list()
    title = song['title']  # song title
    lyrics = song['lyrics']  # song lyrics
    songInfo.append((title,lyrics))
    artist_dict[title] = songInfo  # assign list to song dictionary entry named after song title
    return print('Json Search Complete for',title,'...')

def updateCSV_file(file):
    upload_count = 0  # Set upload counter
    with open(file, 'w', newline='', encoding='utf-8') as file:  # open a new csv file
        a = csv.writer(file, delimiter=',')  # split by comma
        headers = ["Title","Lyrics"]
        a.writerow(headers)  # add header row
        for song in artist_dict:
            a.writerow(artist_dict[song][0])
            upload_count += 1

        #print(str(upload_count) + " songs have been uploaded")
    return print('CSV Updated...')

def search_csv_file(file,currentObject):
    with open(file) as f_obj:
        reader = csv.reader(f_obj, delimiter=',')
        for line in reader:  # Iterates through the rows of your csv
            allSongs = line[1].splitlines() #split entire song into new lines to be read 1 by 1
            for i in range(len(allSongs)): #loop through each line
                if currentObject in allSongs[i]: #searching for object
                    print('Song: ',line[0],'| Lyrics: ',allSongs[i])
                    keyName = line[0] +'_'+currentObject+'_'+str(i) #setting key to be unqiue
                    lyricsFound[keyName.upper()] = allSongs[i] #setting key and setting it to uppercase personal preference
                    allLyrics.update(lyricsFound) #updating master dict
    return print('Object Search Completed...')

def main(api, artist, maxSongs, sortBy, currentObject):
    print('Lyrics Search started...')
    search = api.search_artist(artist, max_songs=maxSongs, sort=sortBy)

    artistFileName = artist.replace(" ", "")
    artistFile = 'Lyrics_' + artistFileName + '.json'
    search.save_lyrics(artistFile)
    Artist = json.load(open(artistFile))
    file = "outputLyrics/lyricsGeniusFile.csv"

    for i in range(maxSongs):
        collectSongData(Artist['songs'][i])
    updateCSV_file(file)
    search_csv_file(file,currentObject)
    os.remove(artistFile) #removing the artist file to remove the (y/n) promote for overriding the file
    print('Lyrics Search ended...')

if __name__ == '__main__':
    main()
