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
TOPIC = 0
BIN_VECTOR = 2
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
    i = len(documents) + 1
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
    for doc in documents:
        word_list =  documents[doc][WORD_LIST]
        vector = [0]*len(vocabulary)
        for word,count in word_list:
            if word in vocabulary:
                vector[voc.index(word)] =1
        documents[doc][BIN_VECTOR] = vector
    return documents

def hd(vector_train, vector_test):
    size =len(vector_train)
    count =0
    for i in range(0, size):
        # if vector_train[i] != vector_test[i]:
        #     count += 1
        count+=(vector_train[i] ^ vector_test[i])

    return count

def euclidean_distance(instance1, instance2):
    distance = 0.0
    for i in range(len(instance1)):
        distance += (instance1[i] - instance2[i])**2
    return np.sqrt(distance)
    
def hamming_prediction(train, test, n_neighbors=1):
    allTestNeighbers=[]
    allPredictedOutputs =[]

    allDistances = []
    size = len(test)
    half = size/2
    quarter =  half/2
    third_quarter = 3*quarter
    i =0 
    for instance in test:
        for vector in train:
            distance = hd(train[vector][BIN_VECTOR], test[instance][BIN_VECTOR])
            allDistances.append((vector, train[vector][TOPIC], distance))

        allDistances.sort(key=lambda x: x[2])
        voteCount = np.zeros(uniqueOutputCount)
        neighbors = []
        for n in range(n_neighbors):
            neighbors.append(allDistances[n][0])
            class_label = topic_list.index(allDistances[n][1])
            voteCount[class_label] += 1
        
        #Determine the Majority Voting (Equal weight considered)
        predictedOutput = np.argmax(voteCount)
        
        allTestNeighbers.append(neighbors)
        allPredictedOutputs.append(topic_list[predictedOutput])
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
            if predictedOutput[doc-1] == document_test[doc][TOPIC]:
                correctCount += 1
        except:
            print (doc)
            break
        
        totalCount += 1
    
    print("Total Correct Count: ",correctCount," Total Wrong Count: ",totalCount-correctCount," Accuracy: ",(correctCount*100)/(totalCount))


# %%
#Following operation is required if you run this cell for the first time
#!pip3 install bs4
from bs4 import BeautifulSoup as bs

train = 500
validate = 700
test = 1200
with open("data/topics.txt") as file:
    for topic in file:
        uniqueOutputCount+=1

        topic = topic.rstrip('\n')
        topic_list.append(topic)
        print(topic)
        with open('data/Training/'+ topic +'.xml','r',encoding='utf-8') as file:
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
predictedOutput,_ = hamming_prediction(document_train, document_test, 3)
#%%
print("len of prediction: ", len(predictedOutput), " Len of test: ", len(document_test))

#%%
performanceEvaluation(predictedOutput, document_test)

print("============ evaluation done ===========\n\n")
print(len(vocabulary))
print(len(document_train))
print(len(document_vaidate))
print(len(document_test))
print("finished")

#%%
hd(train_representation[1][BIN_VECTOR], test_representation[1][BIN_VECTOR])
# %%
