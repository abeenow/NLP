import pickle
import nltk

from os import path
from os.path import sep
from nltk import stem
from string import punctuation
from nltk.corpus import stopwords
from classlib import Review

def load_classifier(review,prod_type,class_type):
	
	file_name = "pickle"+sep+review.pick_names[int(prod_type)][class_type]+".pickle"
	if(path.isfile(file_name) == False):
		if(class_type == 0):
			review.write_naive_bayes_classifiers(int(prod_type))
		elif(class_type == 1):
			review.write_maxent_classifiers(int(prod_type))
			
	f = open(file_name)
	classifier = pickle.load(f)
	f.close()
	return classifier
	
def get_rating(review,prod_type,classifier):
	
	## TODO Lemmatize the review, convert into features Get count for each word just like feats
	rating=0.0
	feats = {}
	stopset = set(stopwords.words('english'))
	for word in nltk.word_tokenize(review):
		if ( word not in stopset and word not in punctuation ):
			if(feats.has_key(word)) :
				feats[word] = feats[word] + 1
			else:
				feats[word] = 1
				
	## Now Using the classifier, classify and get the rating
	rating = classifier.classify(feats)
	return rating
	
def main():
	review = Review()
	while True :
		print("Select Amazon product type:")
		prod_type = raw_input("\n\t1. Books\n\t2. DVD\n\t3. Electronics\t\n\t4. Kitchen\n\t5. Exit\n\n\t")
		
		if(int(prod_type) == 5):
			print("Quitting...")
			break;
		review_string = raw_input("Provide your review here: ")
		classifier=load_classifier(review,prod_type,0)
		rating = get_rating(review_string,prod_type,classifier)
		print("Rating using the Naive Bayes classification method: %.1f" %rating)
		
		classifier=load_classifier(review,prod_type,1)
		rating = get_rating(review_string,prod_type,classifier)
		print("Rating using the Maximum Entropy classification method: %.1f" %rating)
		
		
		choice = raw_input("Do you wish to Continue? [Y/N]")
		if (choice.lower() == 'n'):
			print("Quitting...")
			break;		

if __name__ == '__main__':
	main()
