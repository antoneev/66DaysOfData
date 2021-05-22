import json
import csv
import shutil
import re

# Create an empty dictionary to store your songs and related data
artist_dict = {}
allLyrics = {}
allSongTitles = []
file = "outputLyrics/lyricsGeniusFile.csv"  # File Path

# Creating JSON file from search
def findArtistSongs(api, artist, maxSongs, sortBy):
    try:
        #search = api.search_artist(artist, max_songs=maxSongs, sort=sortBy) # Select songs based on ASC order of song name
        search = api.search_artist(artist, max_songs=maxSongs) # Random selection of songs

        artistFileName = re.sub(r'[^A-Za-z0-9]+', '', artist) # Removing all alphanumeric characters from string
        artistFile = 'Lyrics_' + artistFileName + '.json' # Lyrics file name used instead of default to ensure consistancy of file names when weird characters used
        search.save_lyrics(artistFile,overwrite=True) # Creation JSON file overwrite=True overides JSON with same name
        shutil.move(artistFile, "outputLyrics/"+artistFile) # Moving file as a personal perference so individuals can see JSON on git rather than deleting it
        print('JSON Created ...')

        Artist = json.load(open("outputLyrics/" + artistFile))  # Loading JSON file

        # Looping through each song while calling the collectSongData function
        for i in range(maxSongs):
            collectSongData(Artist['songs'][i])
        updateCSV_file(file)  # Updating CSV calling updateCSV_file function

        return artistFile
    except:
        artistFile = 'Timeout: Request timed out: HTTPSConnectionPool'
        return artistFile

# Searching JSON file to collect song information
def collectSongData(song):
    songInfo = list()
    title = song['title']  # Collects song title
    lyrics = song['lyrics']  # Collets song lyrics
    songInfo.append((title,lyrics))
    artist_dict[title] = songInfo  # Assign list to song dictionary entry named after song title
    if title not in allSongTitles:
        allSongTitles.append(title)
    return print('JSON Search completed for',title,'...') # Indicates JSON search completed

# Updating CSV File
def updateCSV_file(file):
    upload_count = 0  # Set upload counter
    with open(file, 'w', newline='', encoding='utf-8') as file:  # Open a new csv file
        a = csv.writer(file, delimiter=',')  # Split by comma
        headers = ["Title","Lyrics"] # Read headers
        a.writerow(headers)  # Add header row
        for song in artist_dict:
            a.writerow(artist_dict[song][0]) # Write to CSV
            upload_count += 1 # Update Count
        #print(str(upload_count) + " songs have been uploaded")
    return print('CSV Updated...') # Indicates CSV update completed

# Search CSV for Elements in Lyrics
def search_csv_file(file,currentElement):
    with open(file) as f_obj:
        reader = csv.reader(f_obj, delimiter=',')
        for line in reader:  # Iterates through the rows of your csv
            allSongs= line[1].splitlines() # Split entire song into new lines to be read 1 by 1
            for i in range(len(allSongs)):
                if currentElement in allSongs[i].lower(): # Searching for element and making all lowercased for searching
                    print('Song: ',line[0],'| Lyrics: ',allSongs[i])
                    keyName = line[0] +'_'+currentElement+'_'+str(i) # Setting key to be unqiue
                    allLyrics[keyName.upper()] = allSongs[i] # Setting key and setting it to uppercase personal preference
    return print('Element Search completed...') # Indicates element search completed

def main(currentElement):
    print('Lyrics Search started...') # Indicating algorithm started

    search_csv_file(file,currentElement) # Searching for lyrics using the search_csv_file function

    print('Lyrics Search completed...') # Indicating algorithm completed

if __name__ == '__main__':
    main()
