#generate features
#the words in the sms will be our features.
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

#create a list which will contain all words from each message in process words
def createBag(preprocess_messages):
	bag_of_words=[]
	for message in preprocess_messages:
		words=word_tokenize(message)
		for word in words:
			bag_of_words.append(word)
	bag_of_words=nltk.FreqDist(bag_of_words)
	return bag_of_words

#finds feature word in each message of dataset
def featureFinder(message, feature_words):
	#essentially a list with indexes as feature words. Index is true if feature word is present in msg else not
	features_in_a_msg={}
	w_tokens=word_tokenize(message)
	for fw in feature_words:
		features_in_a_msg[fw]=(fw in w_tokens)
	return features_in_a_msg

def featureSetGenerator(feature_words, zip_messages):
	#np.random.seed = 1
	#np.random.shuffle(zip_messages)
	feature_sets = [(featureFinder(text, feature_words), label) for (text, label) in zip_messages]
	return feature_sets