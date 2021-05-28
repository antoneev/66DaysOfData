import sqlite3
connection = sqlite3.connect("db/data.db")
cursor = connection.cursor()

def createTable():
        sql_file = open("db/database.sql")
        sql_as_string = sql_file.read()
        cursor.executescript(sql_as_string)

def searchForWords(currentWord):
        cursor.execute("SELECT * FROM keywords WHERE word='"+currentWord+"'")
        row = cursor.fetchone()
        if row == None:
                result = False
                return result
        else:
                result = True
                return result


