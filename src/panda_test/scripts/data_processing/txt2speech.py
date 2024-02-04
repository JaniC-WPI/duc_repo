from gtts import gTTS
import os

# Load your text file
with open('/home/jc-merlab/Documents/Ethics_Class/AI_Ethics_Text_Book/Chapter_1.txt', 'r') as file:
    text = file.read()

# Convert to speech
tts = gTTS(text=text, lang='en')
tts.save("/home/jc-merlab/Documents/Ethics_Class/AI_Ethics_Text_Book/Chapter_1.mp3")