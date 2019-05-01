#contains various sklearn classifiers as ensemble model
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
import pandas as pd

def ensembleClassification(training, testing):
	names = [
		"K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
		"Naive Bayes", "SVM Linear"
	]

	classifiers = [
		KNeighborsClassifier(),
		DecisionTreeClassifier(),
		RandomForestClassifier(),
		LogisticRegression(),
		SGDClassifier(),
		MultinomialNB(),
		SVC(kernel = 'linear')
	]

	zip_models=zip(names, classifiers)
	ensemble_model_nltk=SklearnClassifier(VotingClassifier(estimators=list(zip_models), voting='hard'))
	ensemble_model_nltk.train(training)
	return ensemble_model_nltk

def print_accuracy_metrics(ensemble_model_nltk, testing):
	accuracy=nltk.classify.accuracy(ensemble_model_nltk, testing)*100
	print("Ensemble model accuracy: ", accuracy)

	msg_features, labels=zip(*testing)
	prediction = ensemble_model_nltk.classify_many(msg_features)
	print("Confusion Matrix: ")
	accuracy_df=pd.DataFrame(
					    confusion_matrix(labels, prediction),
					    index = [['actual', 'actual'], ['ham', 'spam']],
					    columns = [['predicted', 'predicted'], ['ham', 'spam']]
			    )
	print(accuracy_df)

