import youtube_dl, pysrt
import numpy as np
import urllib
import glob
import os
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import model_from_json

config_save="model_word"

class audio_source(object):
    def __init__(self, url):
        self.url = url
        self.ydl_opts = {
            'subtitleslangs': ['fr'],
            'writesubtitles': True,
            'writeautomaticsub': True}
        self.subtitlesavailable = self.are_subs_available()
        if self.subtitlesavailable:
            self.grab_auto_subs()
    def are_subs_available(self):
        with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
            subs = ydl.extract_info(self.url, download=False)
            if 'requested_subtitles' in subs:
                self.title = subs['title']
                self.subs_url = subs['requested_subtitles']['fr']['url']
                print(self.title+" "+self.subs_url)
                return True
            else:
                return False
        return False
    def grab_auto_subs(self):
        try:
            urllib.request.urlretrieve(
                self.subs_url, 'youtube-dl-texts/' + self.title + '.srt')
            print("subtitles saved directly from youtube\n")
            text = pysrt.open('youtube-dl-texts/' + self.title + '.srt')
            self.text = text.text.replace('\n', ' ')
        except IOError:
            print("\n *** saving sub's didn't work *** \n")
    def loadurls():
        with open('other/url_list','r') as datafile:
            url_list = datafile.read().splitlines()
        total_text = []
        for u in url_list:
            try:
                total_text.append(audio_source(url=u).text)
            except AttributeError:
                pass
        total_text = ' '.join(total_text).lower()
        print(len(total_text))
def test():
    text=''
    for srt in glob.glob("youtube-dl-texts/*.srt"):
        tmp=pysrt.open(srt)
        text += tmp.text
    return text.replace('\n', ' ')
def TrainingData():
    if os.path.isfile(config_save): 
        data = pickle.load( open( config_save, "rb" ) )
        return data['t'],data['c'],data['ci'],data['ic']
    return TrainingDataFromTXT()
def TrainingDataFromTXT():
    total_text=test().split()
    chars = set(total_text)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    pickle.dump({"t":total_text,"c":chars,"ci":char_indices,"ic":indices_char},open(config_save,"wb"))
    return total_text,chars,char_indices,indices_char
def SaveModel(model):
    print('Sauvegarde du modele')
    model_json = model.to_json()
    with open(config_save+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(config_save+".h5")
    return model
def TrainModel():
    if os.path.isfile(config_save+'.json'):
        print('Récupération du modele...') 
        json_file = open(config_save+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(config_save+".h5")
        return loaded_model
    return TrainModel_Data()
def TrainModel_Data():
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(total_text) - maxlen, step):
        sentences.append(total_text[i: i + maxlen])
        next_chars.append(total_text[i + maxlen])
    print('nb sequences:', len(sentences))
    print('nb ci:', len(char_indices))
    print('nb ic:', len(indices_char))

    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    #AI
    model = Sequential()
    #model.add(LSTM(len(chars), 512, return_sequences=True))
    model.add(LSTM(len(chars), input_length=maxlen, input_dim=len(char_indices), return_sequences=True))
    model.add(Dropout(0.20))
    # use 20% dropout on all LSTM layers: http://arxiv.org/abs/1312.4569

    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.20))

    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.20))

    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.20))

    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    # compile or load weights then compile depending
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(X,y,nb_epoch=50)
    return SaveModel(model)
def gentext(text):
    seed_text = text.split()
    while len(seed_text)<4:
    	seed_text.insert(0,'!')
    generated = '' + text
    print('------'*5+'\ndemande: \n'+'"' + text +'"')
    print('------'*5+'\n generation...\n'+ '------'*5)
    for iteration in range(22):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(seed_text):
            x[0, t, char_indices[char]] = 1.
        preds = model.predict(x, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]
        generated += ' '+next_char
        seed_text.append(next_char)
        seed_text = seed_text[1:]
    print('\n\nphrase générée: ' + generated)
maxlen = 5
txt_fol="le ministère de la culture"
#txt_fol="Elle fait à peu près"
#txt_fol="On va lui faire des"
#txt_fol="Bonjour"
#txt_fol="Bonjour à toutes et à"
#txt_fol="ouais d'accord mais c'est chez"
total_text,chars,char_indices,indices_char = TrainingData()
model=TrainModel()
gentext(txt_fol)