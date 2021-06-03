# Article Assistance 📑
## [View Deployed App 🚀](https://share.streamlit.io/antoneev/66daysofdata/main/articleAssistance/app.py)

# Summary of Project 📖
## Current Problem 🥲

Newspapers currently produce papers above the reading level of the average American. It is reported that the average American reads at a 7th- to 8th-grade level. While newspapers are reported to be written at a 9th- to 15th-grade level, this leaves a huge disconnect as portions of society are unable to read articles. <br> Sources: [Reading Level](https://www.wyliecomm.com/2020/11/whats-the-latest-u-s-literacy-rate/#:~:text=The%20average%20American%20reads%20at%20the%207th%2D%20to%208th%2Dgrade,for%20Disease%20Control%20and%20Prevention) [Newspaper Level](http://www.impact-information.com/impactinfo/newsletter/plwork15.htm)

## Stakeholders  👨‍👩‍👧‍👦

Anyone who may need assistance reading and/or breaking down complex words.

## Solution 🥳

This application aims to assist readers by finding and extracting complex words while displaying the words pronunciation and definitions in a pdf file.

# Challenges Found when conducting this Project 🥲

There is no dataset or package which contains "complex words". Therefore, I aimed to gather all words individuals should know by 8th grade. However, this list still missed different words. Thus, I attempted to assist this list by using the syllable count of a word.

# App Images 📷
![](imgs/example.png?raw=true)

# Future Work 🔮

* Conducting this application with a better list of words.
* Adding the ability to extract from .png, .jpg and other image files

# Resources 🗃️
* [SQLite3](https://www.sqlite.org/index.html)
* [Streamlit](https://streamlit.io/)
* [k- to 8th-grade Words](https://www.spelling-words-well.com/free-preschool-games.html)
* [PDF to Text](https://pypi.org/project/slate3k/)
* [Creating PDFs using reportlab](http://theautomatic.net/2021/03/25/how-to-create-pdf-files-with-python/)
* [Google Dictionary API](https://dictionaryapi.dev/)
* [Embedding PDF into Streamlit](https://blog.jcharistech.com/2020/11/30/how-to-embed-pdf-in-streamlit-apps/)