import nltk
import pickle
import glob

from os.path import sep

class Review:
	
	def __init__(self):
		
		self.pick_names = {1:['Amazon_Book_NB','Amazon_Book_ME'],2:['Amazon_DVD_NB','Amazon_DVD_ME'],3:['Amazon_Electronics_NB','Amazon_Electronics_ME'],4:['Amazon_Kitchen_NB','Amazon_Kitchen_ME']}
		
		if (self.check_pickles() == False):
			# Start reading the Dataset here
			self.feats = {1:self.get_features("books"),2:self.get_features("dvd"),3:self.get_features("electronics"),4:self.get_features("kitchen")}
				
	def get_features(self,product_type):
		
		data = open("Data"+sep+product_type+sep+"train")
		feats=[]
		for line in data.readlines():
			tokens = line.split(" ")
			d = {}
			label_val = tokens[-1].split(":")
			for words in tokens[:-1]:
				key_val = words.split(":")
				d[key_val[0]] = float(key_val[1])
			feats.append((d,float(label_val[1])))
		
		return feats
		
	def check_pickles(self):
		value = True
		if (len(glob.glob1("pickle","*.pickle")) != 8) :
			value = False
		return value
		
	def write_to_pickle(self,classifier,file_name):
		
		# Save the classifiers to the pickle as binary files
		f = open("pickle"+sep+file_name+".pickle", 'wb')
		pickle.dump(classifier, f)
		f.close()
		
	def write_naive_bayes_classifiers(self,option):
		
		# Train the Naive Bayes classifier with the extracted features and write to pickle for future use
		self.write_to_pickle(nltk.NaiveBayesClassifier.train(self.feats[option]),self.pick_names[option][0])
		
	def write_maxent_classifiers(self,option):	
		
		# Train the MaxEntropy classifier with the extracted features and write to pickle for future use
		self.write_to_pickle(nltk.MaxentClassifier.train(self.feats[option], 'gis',max_iter=3),self.pick_names[option][1])
