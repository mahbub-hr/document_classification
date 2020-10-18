# %%
import numpy as np
import string
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter

WORD_LIST = 1
MAX = 99999999
TOPIC = 0
BIN_VECTOR = 2
HAMMING = 1
EUCLIDEAN = 2
TF_IDF = 3

vocabulary = set()
topic_list =[]
document_train = {}
document_test = {}
document_vaidate = {}
uniqueOutputCount=0

# %%
def text_process(text):

    text = text.lower()
    # print("\n===After Lowercase:===\n", text)

    #Number Removal
    import re
    text = re.sub(r'[-+]?\d+', '', text)
    # print("\n===After Removing Numbers:===\n", text)

    #Remove punctuations
    text=text.translate((str.maketrans('','',string.punctuation)))
    # print("\n===After Removing Punctuations:===\n", text)

    #Tokenize
    text = word_tokenize(text)
    # print("\n===After Tokenizing:===\n", text)

    #Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if not word in stop_words]
    # print("\n===After Stopword Removal:===\n", text)

    #Lemmatize tokens
    lemmatizer=WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    # print("\n===After Lemmatization:===\n", text)

    #Stemming tokens
    stemmer= PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    # print("\n===After Stemming:===\n", text)
    counts = Counter(text)
    return list(counts.items())

def preprocess(rows, topic, documents, add_to_voc=False):
    i = len(documents)
    for item in rows:
        body = bs(item.get('body'),'lxml').get_text().encode('ascii','ignore').decode('ascii')
        document = {}
        body = text_process(body)
        # print(body)
        document[WORD_LIST] = body
        if add_to_voc:
            vocabulary.update([a for a,b in body])
        document[TOPIC] = topic
        documents[i] = document
        i += 1

def generate_binary_vector(voc, documents):
    vector_representation = []
    for doc in documents:
        word_list =  documents[doc][WORD_LIST]
        vector = [0]*len(vocabulary)
        for word,count in word_list:
            if word in vocabulary:
                vector[voc.index(word)] = count
        vector_representation.append(vector)
    return np.array(vector_representation)

def hd(test, train, prevMaxMin = None):
    return np.sum(np.logical_xor(test, train))

def euclidean_distance(test, train, prevMaxMin = None):
    # distance = 0.0
    # for i in range(len(instance1)):
    #     distance += (instance1[i] - instance2[i])**2
    #     # if prevMaxMin is not None and distance > prevMaxMin:
    #     #     return np.sqrt(distance)
    # return np.sqrt(distance)
    return np.linalg.norm(test-train)
def cos_similarity(test, train, prevMaxMin = None):
    pass

def tf_idf(prevMaxMin = None):
    pass
def prediction(train, test, n_neighbors=1, method=1):
    allTestNeighbers=[]
    allPredictedOutputs =[]

    allDistances = []
    size = len(test)
    half = size/2
    quarter =  int(half/2)
    third_quarter = 3*quarter

    i =0 
    prevMaxMin = MAX
    for r_test in test:
        j =0
        for r_train in train:
            distance = METHOD[method](r_test, r_train) 
            allDistances.append((j, document_train[j][TOPIC], distance))
            j+=1

        allDistances.sort(key=lambda x: x[2])
        voteCount = {}
        # neighbors = []
        for n in range(n_neighbors):
            # neighbors.append(allDistances[n][0])
            class_label = allDistances[n][1]
            if class_label not in voteCount:
                voteCount[class_label] = 1
            else: 
                voteCount[class_label] += 1
        
        #Determine the Majority Voting (Equal weight considered)
        predictedOutput = max(voteCount, key=voteCount.get)
        
        # allTestNeighbers.append(neighbors)
        allPredictedOutputs.append(predictedOutput)
        i+= 1
        if i == half:
            print("Half of the test is done\n")
        elif i == size:
            print("All done")
        
    return allPredictedOutputs, allTestNeighbers

def performanceEvaluation(predictedOutput, document_test):
    totalCount = 0
    correctCount = 0
    
    for doc in document_test:
        try:
            if predictedOutput[doc] == document_test[doc][TOPIC]:
                correctCount += 1
        except:
            print (doc)
            break
        
        totalCount += 1
    accuracy = round((correctCount*100)/(totalCount),2)
    print("Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",)
    return accuracy

# %%
#Following operation is required if you run this cell for the first time
#!pip3 install bs4
from bs4 import BeautifulSoup as bs
METHOD = {
    1: hd,
    2: euclidean_distance,
    3: tf_idf,
}
train = 100
validate = 150
test = 250
with open("Data/topics.txt") as file:
    for topic in file:
        uniqueOutputCount+=1

        topic = topic.rstrip('\n')
        topic_list.append(topic)
        print(topic)
        with open('Data/Training/'+ topic +'.xml','r',encoding='utf-8') as file:
            content = file.read()
            # topic = "Anime"
            soup = bs(content,features="lxml")
            rows = soup.findAll("row")
            preprocess(rows[:train], topic, document_train, add_to_voc=True)
            preprocess(rows[train:validate], topic, document_vaidate)
            preprocess(rows[validate:test], topic, document_test)

voc = list(vocabulary)
print("============ creating binary representation ==========\n")
train_representation = generate_binary_vector(voc, document_train)
validate_representaion = generate_binary_vector(voc, document_vaidate)
test_representation = generate_binary_vector(voc, document_test)
print("============ representation is ready ==========\n")

#%%
n_neighbors = [1,3,5]
best_k = 1
max_accuracy = 0
for k in n_neighbors:
    predictedOutput,_ = prediction(train_representation, validate_representaion, k, HAMMING)
    accuracy = performanceEvaluation(predictedOutput, document_vaidate)
    print("k: ", k, " accuracy: ", accuracy)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_k = k

print("============ evaluation done ===========\n\n")

print(len(vocabulary))
print(len(document_train))
print(len(document_vaidate))
print(len(document_test))
print("finished")
