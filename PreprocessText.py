#Loads data and preprocesses it
import nltk
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

def loadData(file_name):
	df=pd.read_table(file_name, header=None, encoding='utf-8')
	return df

def encodeMessages(df):
	classes=df[0]
	encoder=LabelEncoder()
	binary_labels=encoder.fit_transform(classes)
	return binary_labels

def preprocessMessages(df):
	text_messages=df[1]
	# Replace email addresses with 'email'
	process_messages = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')

	# Replace URLs with 'webaddress'
	process_messages = process_messages.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')

	# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
	process_messages = process_messages.str.replace(r'£|\$', 'moneysymb')
	    
	# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
	process_messages = process_messages.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumbr')
	    
	# Replace numbers with 'numbr'
	process_messages = process_messages.str.replace(r'\d+(\.\d+)?', 'numbr')

	# Remove punctuation
	process_messages = process_messages.str.replace(r'[^\w\d\s]', ' ')

	# Replace whitespace between terms with a single space
	process_messages = process_messages.str.replace(r'\s+', ' ')

	# Remove leading and trailing whitespace
	process_messages = process_messages.str.replace(r'^\s+|\s+?$', '')
	
	#lowercase
	process_messages = process_messages.str.lower()

	#remove stop words
	stop_words=set(stopwords.words('english'))
	process_messages=process_messages.apply(lambda x:' '.join(term for term in x.split() if term not in stop_words))

	#use stemming
	ps=nltk.PorterStemmer()
	process_messages=process_messages.apply(lambda x:' '.join(ps.stem(term) for term in x.split()))

	return process_messages


