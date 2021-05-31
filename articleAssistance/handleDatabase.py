import sqlite3

# Creating DB connection
connection = sqlite3.connect("db/data.db")
cursor = connection.cursor()

# Creating table
def createTable():
        sql_file = open("db/database.sql")
        sql_as_string = sql_file.read()
        cursor.executescript(sql_as_string)

# Searching for words
def searchForWords(currentWord):
        # Searching for each word in article
        cursor.execute("SELECT * FROM keywords WHERE word='"+currentWord+"'")
        # Checking if a word was found
        row = cursor.fetchone()
        # Returning results to the backend
        if row == None:
                result = False
                return result
        else:
                result = True
                return result


