import sqlite3

# Creating DB connection
connection = sqlite3.connect("data.db")
cursor = connection.cursor()

cursor.execute("SELECT * FROM keywords WHERE word='"+'american'+"'")
print(cursor.fetchone())

