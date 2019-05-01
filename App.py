#Main App
from sklearn import model_selection
from PreprocessText import *
from FeatureExtract import *
from Classifiers import *
import nltk

df=loadData('SMSSpamCollection')
binary_labels=encodeMessages(df)
preprocess_messages=preprocessMessages(df)

#create bag of words from preporcess messages by tokenizing
bag_of_words=createBag(preprocess_messages)

#take the 500 most common words as feature words
f=list(bag_of_words.most_common(250))
feature_words=[]
for fw in f:
	feature_words.append(fw[0])

#print(preprocess_messages[0])
#features=featureFinder(preprocess_messages[0], feature_words)
#for key, value in features.items():
#    if value == True:
#        print(key)


#pass these feature words to a function that will find feature words in a message and return it along with its label
zip_messages=zip(preprocess_messages, binary_labels)
feature_sets=featureSetGenerator(feature_words, zip_messages)
training, testing = model_selection.train_test_split(feature_sets, test_size = 0.25, random_state=1)
print("training dataset length: ",len(training))
print("testing dataset length: ",len(testing))

#train an ensemble model
ensemble_model_nltk=ensembleClassification(training, testing)

#print its accuracy metrics
print_accuracy_metrics(ensemble_model_nltk, testing)