#Main App
from sklearn import model_selection
from PreprocessText import *
from FeatureExtract import *

df=loadData('SMSSpamCollection')
binary_labels=encodeMessages(df)
preprocess_messages=preprocessMessages(df)
#create bag of words from preporcess messages by tokenizing
bag_of_words=createBag(preprocess_messages)
#take the 1500 most common words as feature words
feature_words=list(bag_of_words.keys())[:1500]
#pass these feature words to a function that will find feature words in a ms=essage and return it along with its label
zip_messages=zip(preprocess_messages, binary_labels)
feature_sets=featureSetGenerator(feature_words, zip_messages)

training, testing = model_selection.train_test_split(feature_sets, test_size = 0.25, random_state=1)
print(len(training))
print(len(testing))
