# Caption Suggestions Using Lyrics 🎶

## [View Deployed App 🚀](https://share.streamlit.io/antoneev/66daysofdata/main/captionSuggestionsUsingLyrics/app.py)
## [Read Medium Post 📖](https://antoneevansjr.medium.com/caption-suggestions-using-lyrics-e4142dd2e0f7)

## Summary of Project 📝
### Current Problem 🥲
The current problem faced by many are not having witting, insightful or simply great captions. With the rise of social media, everything needs a caption. From a photo with your grandmother to the cute photo with your dog, and even the photo with you “painting the town red”, but the issue is most of us don’t have the right words to caption our photos, this makes us wait weeks before posting, and sometimes we never post for this reason. Other times it’s simply posted without a caption.

### Stakeholders 👨‍👩‍👧‍👦
The stakeholder for this project is anyone who uses social media. More particular those who prefer to use simply 1-line captions.

### Solution 🥳
The solution to this issue is to use image detection to suggest song lyrics based on common elements in the image which can be used as a caption.
This is done as followed:
1.	Object and color detection are used to find common items and colors in the photo.
2.	Use similar word suggestion to suggest similar words to the objects found. This is done as “boat” may not be in a song but “yacht” maybe.
3.	Implement the lyricgenius package which returns JSONs of songs and their lyrics. 
4.	Searches through these lyrics to find lines where the element may be mentioned.
5.	Streamlit was used to wrap the app.

## Challenges Found when conducting this Project 🥲
1.	Lyricgenius documentation is not the best. I was unable to find all the “sortby=” options; therefore, I did not use the option. As sortby=title gives all songs in ASC order by title. However, without the sortby= option users are given randomly selected songs.
2.	cv2 had an issue loading in Jupyter notebook so I used PyCharm.
3.	There seemed to be limited documentation on color detection when wanting to load a photo and return all colors found. Therefore, I had to use a combination of different resources. 
4. Streamlit runs in real time. Therefore, buttons and other elements must have if-else logic tied to it, so the system does not excute. This makes having nested elements (buttons, etc.) a bit tricky. 
5. Streamlit caches information which is great in some instances. However, this causes elements from different uploaded images to be appended together. For example, if you upload image 1 followed by image 2. Image 2 elements were being appended to those of image 1. This caused the output to be incorrect. Therefore, a clear was done of all elements at the end of the final output to the screen.

## Performance 💨
Note: times do fluctuate
* If no objects are found (only colors found) – Runtime with Lyrics 17 seconds
* Objects and Colors found – Runtime 23 seconds

## App Images 📷

![](imgs/top.png?raw=true)
![](imgs/middle.png?raw=true)
![](imgs/bottom.png?raw=true)


## Files within the files folder which is not available within this repo 🗄️
* [Colors.csv]( https://github.com/codebrainz/color-names/blob/master/output/colors.csv)
* [Coco.names]( https://github.com/pjreddie/darknet/blob/master/data/coco.names)
* [frozen_inference_graph.pb]( https://github.com/datitran/object_detector_app/blob/master/object_detection/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb)
* [glove.6B.300d]( https://www.kaggle.com/thanakomsn/glove6b300dtxt)
* [ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt]( https://gist.github.com/dkurt/54a8e8b51beb3bd3f770b79e56927bd7)

## Resources 🗃️
* [Object Detection video 1](https://www.youtube.com/watch?v=HXDD7-EnGBY)
* [Object Detection video 2]( https://www.youtube.com/watch?v=RFqvTmEFtOE)
* [Color Detection Colors.csv]( https://towardsdatascience.com/building-a-color-recognizer-in-python-4783dfc72456)
* [LyricGenius GitHub](https://github.com/johnwmillr/LyricsGenius)
* [LyricGenius Download Lyrics]( https://rareloot.medium.com/how-to-download-an-artists-lyrics-from-genius-com-using-python-984d298951c6)
* [Streamlit](https://docs.streamlit.io/en/stable/)