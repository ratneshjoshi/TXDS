from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, Adadelta
from keras.models import model_from_json
from keras import backend as K
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

from sklearn.metrics import classification_report, confusion_matrix
from textblob import TextBlob

import numpy as np, string, pickle, warnings, random
import matplotlib.pyplot as plt

# from IPython.display import display, HTML

from deepexplain.tensorflow import DeepExplain
import tensorflow as tf
warnings.filterwarnings("ignore")

topWords = 50000
maxWords = 200
nb_classes = 2
imdbDataPicklePath = './data/imdbData.pickle'
downloadFlag = 0

if downloadFlag == 0:

    # Downloading data
    imdbData = imdb.load_data(path='imdb.npz', num_words=topWords)

    # Pickle Data
    with open(imdbDataPicklePath, 'wb') as handle:
        pickle.dump(imdbData, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(imdbDataPicklePath, 'rb') as pHandle:
    imdbData = pickle.load(pHandle)
    
(x_train, y_train), (x_test, y_test) = imdbData

stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", \
             "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", \
             'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', \
             'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', \
             'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
             'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', \
             'above', 'below', 'to', 'from', 'off', 'over', 'then', 'here', 'there', 'when', 'where', 'why', \
             'how', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'own', 'same', 'so', \
             'than', 'too', 's', 't', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
             've', 'y', 'ma']
word2Index = imdb.get_word_index()
index2Word = {v: k for k, v in word2Index.items()}
index2Word[0] = ""
sentimentDict = {0: 'Negative', 1: 'Positive'}

def getWordsFromIndexList(indexList):
    wordList = []
    for index in indexList:
        wordList.append(index2Word[index])

    return " ".join(wordList)

def getSentiment(predictArray):
    pred = int(predictArray[0])
    return sentimentDict[pred]

def getIndexFromWordList(wordList):
    indexList = []
    for word in wordList:
        print(word)
        indexList.append(str(word2Index[word]))
        
    return indexList

print (len(word2Index))

stopIndexList = []

for stopWord in stopWords:
    stopIndexList.append(word2Index[stopWord])

trainData = []

for indexList in x_train:
    processedList = [index for index in indexList]# if index not in stopIndexList]
    trainData.append(processedList)
    
x_train = trainData

'''
Padding data to keep vectors of same size
If size < 100 then it will be padded, else it will be cropped
'''
trainX = pad_sequences(x_train, maxlen = maxWords, value = 0.)
testX = pad_sequences(x_test, maxlen = maxWords, value = 0.)

'''
One-hot encoding for the classes
'''
trainY = np_utils.to_categorical(y_train, num_classes = nb_classes)
testY = np_utils.to_categorical(y_test, num_classes = nb_classes)

p_W, p_U, weight_decay = 0, 0, 0
regularizer = l2(weight_decay) if weight_decay else None
sgdOptimizer = 'adam'
lossFun='categorical_crossentropy'
batchSize=25
numEpochs = 20
numHiddenNodes = 256

model = Sequential()
model.add(Embedding(topWords, numHiddenNodes, name='embedding_layer'))

model.add(Bidirectional(LSTM(numHiddenNodes, return_sequences=True,
                            recurrent_regularizer=regularizer, kernel_regularizer=regularizer,
                            bias_regularizer=regularizer, recurrent_dropout=p_W, dropout=p_U),
                        merge_mode='concat', name='bidi_lstm_layer'))
model.add(LSTM(numHiddenNodes, name = 'lstm_layer'))
model.add(Dropout(0.5, name = 'dropout'))

model.add(Dense(numHiddenNodes, name='dense_1'))
model.add(Dense(nb_classes, name='dense_2'))
model.add(Activation("softmax"))
# adam = Adam(lr=0.0001)
# adadelta = Adadelta(lr=0.01, rho=0.95, epsilon=1e-08)
model.compile(loss=lossFun, optimizer=sgdOptimizer, metrics=["accuracy"])
print(model.summary())

model.fit(trainX, trainY, batch_size=batchSize, epochs=numEpochs, verbose=1, validation_data=(testX, testY))

score = model.evaluate(testX, testY, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

predY = model.predict_classes(testX)
yPred = np_utils.to_categorical(predY, num_classes = nb_classes)
print("Classification Report:\n")
print(classification_report(testY, yPred))

'''
Serialize model to JSON
'''
model_json = model.to_json()
with open("models/imdb_bi_lstm_big_tensorflow_model_ALL.json", "w") as json_file:
    json_file.write(model_json)

'''
Serialize weights to HDF5
'''
model.save_weights("models/imdb_bi_lstm_big_tensorflow_model_ALL.h5", overwrite=True)
print("Saved model to disk...")

'''
Load json and create model
'''
modelNum=1
if modelNum is 1:
    json_file = open('models/imdb_bi_lstm_big_tensorflow_model.json', 'r')
elif modelNum is 2:
    json_file = open('models/imdb_bi_lstm_big_tensorflow_model_NOT.json', 'r')
elif modelNum is 3:
    json_file = open('models/imdb_bi_lstm_big_tensorflow_model_ALL.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

'''
Load weights into new model
'''
if modelNum is 1:
    loaded_model.load_weights("models/imdb_bi_lstm_big_tensorflow_model.h5")
elif modelNum is 2:
    loaded_model.load_weights("models/imdb_bi_lstm_big_tensorflow_model_NOT.h5")
elif modelNum is 3:
    loaded_model.load_weights("models/imdb_bi_lstm_big_tensorflow_model_ALL.h5")

print ("Loading model from disk...")
model = loaded_model
print(model.summary())

num = 120
num_next = num + 1
print("Testing for test case..." + str(num))
groundTruth = testY[num]
print(groundTruth)
print(len(testY))
print

sampleX = testX[num:num_next]
predictionClass = model.predict_classes(sampleX, verbose=0)
prediction = np_utils.to_categorical(predictionClass, num_classes = nb_classes)[0]

print("Text: " + str(getWordsFromIndexList(x_test[num-1])))
print("\nPrediction: " + str(getSentiment(predictionClass)))
if np.array_equal(groundTruth,prediction):
    print("\nPrediction is Correct")
else:
    print("\nPrediction is Incorrect")

def getWordsAndRelevances(sampleX, relevances, prediction):

    unknownIndex = 0 # Index of padding
    indexList = np.ndarray.tolist(sampleX)[0]
    wordList = []
    wordRelevanceList = []
    polarityHighThreshold = 0.4
    polarityLowThreshold = 0.0
    
    # Find word-wize relevance and normalize
    wordRelevances = np.sum(relevances, -1)
    wordRelList = np.ndarray.tolist(wordRelevances)[0]
    
    for i in range(maxWords):
        index = indexList[i]
        relevance = wordRelList[i]

        if index is not unknownIndex:
            word = index2Word[index]
            blobText = TextBlob(word)
            polarity = blobText.sentiment.polarity
            prediction = int(prediction)
#             if prediction is 0 and polarity < -0.4 or prediction is 1 and polarity > 0.4:
            if abs(polarity) > polarityHighThreshold:
                if prediction is 0 and polarity < 0 or prediction is 1 and polarity > 0:
                    relevance = abs(polarity)
                elif prediction is 0 and polarity > 0 or prediction is 1 and polarity < 0:
                    relevance = -1 * abs(polarity)
                    
                print(word, polarity)
                print(word, relevance)
            elif abs(polarity) > polarityLowThreshold:
                if relevance < 0 and polarity > 0 or relevance > 0 and polarity < 0:
                    relevance = polarity
                    print("Here...")
                    print(word, polarity)
                    print(word, relevance)

            wordList.append(index2Word[index])
            wordRelevanceList.append(relevance)

    
    normalizedRelevanceList = [float(rel)/max(map(lambda x: abs(x), wordRelevanceList)) for rel in wordRelevanceList]
    
    return wordList, wordRelevanceList, normalizedRelevanceList

def showWordRelevances(wordList, wordRelevanceList, normalizedRelevanceList):

    print("\nWord Relevances:\n")
    print("\nOriginal Relevance:\n")
    for i in range(len(wordList)):
        word = str(wordList[i])
        # originalRelevance = "{:8.2f}".format(wordRelevanceList[i])
        originalRelevance = wordRelevanceList[i]
        print ("\t\t\t" + str(originalRelevance) + "\t" + word)

    print("\nNormalized Relevance:\n")
    for i in range(len(wordList)):
        word = str(wordList[i])
        normalizedRelevance = "{:8.2f}".format(normalizedRelevanceList[i])
        print ("\t\t\t" + str(normalizedRelevance) + "\t" + word)


def removePunct(inputStr):
    
    # Remove punctuations.
    # punctuations = string.punctuation
    # inputStr = ''.join(ch for ch in inputStr if ch not in punctuations)
    inputStr = inputStr.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    inputList = inputStr.lower().split()
    return inputList

def removeStopWords(inputList):
    processedList = [word for word in inputList if word not in stopWords]
    return processedList

def preProcessQuery(inputStr):
    return removeStopWords(removePunct(inputStr))
#     return removePunct(inputStr)

def getIndexArray(inputStr):
    
    words = preProcessQuery(inputStr)
    # print(words)
    wordIndexList = np.array([word2Index[word] if word in word2Index else 0 for word in words])
    wordIndexArray = np.array([wordIndexList], np.int32)
    wordIndexArray = pad_sequences(wordIndexArray, maxlen = maxWords, value = 0.)
    return wordIndexArray

def getPrediction(inputStr):
    
    wordIndexArray = getIndexArray(inputStr)
    predictionScore = model.predict(wordIndexArray[0:1], verbose=0)
    # print (predictionScore)
    prediction = model.predict_classes(wordIndexArray[0:1], verbose=0)
    return prediction, predictionScore

inputStr = input("Enter a sentence to test polarity: ")
prediction, predictionScore = getPrediction(inputStr)
print("\nPrediction: " + str(getSentiment(prediction)))
print(predictionScore)

with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
    
    '''
    Need to reconstruct the graph in DeepExplain context, using the same weights.
    1. Get the input tensor
    2. Get embedding tensor
    3. Target the output of the last dense layer (pre-softmax)
    '''
    
    inputTensor = model.layers[0].input
    embeddingTensor = model.layers[0].output
    preSoftmax = model.layers[-2].output
    
    # Sample Data for attribution
    wordIndexArray = getIndexArray(inputStr)
    sampleX = pad_sequences(wordIndexArray, maxlen = maxWords, value = 0.)
    
    # Perform Embedding Lookup
    getEmbeddingOutput = K.function([inputTensor],[embeddingTensor])
    embeddingOutput = getEmbeddingOutput([sampleX])[0]
    
    # Get Prediction for attribution
    prediction, predictionScore = getPrediction(inputStr)
    ys = np_utils.to_categorical(prediction, num_classes = nb_classes)
    
    #relevances = de.explain('grad*input', targetTensor * ys, inputTensor, sampleX)
    #relevances = de.explain('saliency', targetTensor * ys, inputTensor, sampleX)
    #relevances = de.explain('intgrad', targetTensor * ys, inputTensor, sampleX)
    #relevances = de.explain('deeplift', targetTensor * ys, inputTensor, sampleX)
    relevances = de.explain('elrp', preSoftmax * ys, embeddingTensor, embeddingOutput)
    #relevances = de.explain('occlusion', targetTensor * ys, inputTensor, sampleX)
    print ("Feature Relevances Generated...")

wordList, wordRelevanceList, normalizedRelevanceList = getWordsAndRelevances(sampleX, relevances, prediction[0])
showWordRelevances(wordList, wordRelevanceList, normalizedRelevanceList)


def rescale_score_by_abs (score, max_score, min_score):
    """
    rescale positive score to the range [0.5, 1.0], negative score to the range [0.0, 0.5],
    using the extremal scores max_score and min_score for normalization
    """
    
    # CASE 1: positive AND negative scores occur --------------------
    if max_score>0 and min_score<0:
    
        if max_score >= abs(min_score):   # deepest color is positive
            if score>=0:
                return 0.5 + 0.5*(score/max_score)
            else:
                return 0.5 - 0.5*(abs(score)/max_score)

        else:                             # deepest color is negative
            if score>=0:
                return 0.5 + 0.5*(score/abs(min_score))
            else:
                return 0.5 - 0.5*(score/min_score)   
    
    # CASE 2: ONLY positive scores occur -----------------------------       
    elif max_score>0 and min_score>=0: 
        if max_score == min_score:
            return 1.0
        else:
            return 0.5 + 0.5*(score/max_score)
    
    # CASE 3: ONLY negative scores occur -----------------------------
    elif max_score<=0 and min_score<0: 
        if max_score == min_score:
            return 0.0
        else:
            return 0.5 - 0.5*(score/min_score)    
  
      
def getRGB (c_tuple):
    return "#%02x%02x%02x"%(int(c_tuple[0]*255), int(c_tuple[1]*255), int(c_tuple[2]*255))

     
def span_word (word, score, colormap):
    return "<span style=\"background-color:"+getRGB(colormap(score))+"\">"+word+"</span>"


def html_heatmap (words, scores, cmap_name="bwr"):
    
    colormap  = plt.get_cmap(cmap_name)
     
    assert len(words)==len(scores)
    max_s     = max(scores)
    min_s     = min(scores)
    
    output_text = ""
    
    for idx, w in enumerate(words):
        score       = rescale_score_by_abs(scores[idx], max_s, min_s)
        output_text = output_text + span_word(w, score, colormap) + " "

    return output_text + "\n"
            
relevanceScores = [-1.00 * scr for scr in wordRelevanceList]
htmlHeatMap = html_heatmap(wordList, relevanceScores)
print ("\nLRP heatmap:")    
display(HTML(htmlHeatMap))