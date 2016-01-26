#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ngrams.py
#  
#  Copyright 2015 Abinav <abinav@localhost>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import argparse
import os.path
import math
import nltk
import string
import sys

from decimal import *


class Ngrams:
	def __init__(self,trainer_path,file_path):
		
		if ( os.path.isfile(trainer_path) and  os.path.isfile(file_path) ):
			self.trainer_path = trainer_path
			self.file_path = file_path
		else:
			print("Invalid file path")
			sys.exit(1)
		# Initialise to empty lists
		self.uni_grams_dict = {}
		self.bi_grams_dict = {}
		# This is used for tokenizing the paragraph into sentences as this is required for probabilty calculation
		self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		
		uni_grams = []
		# Open the training corpus file
		f = open(self.trainer_path,'r')
		for sentence in self.tokenizer.tokenize(f.read()):
			# Add the Start/End tag to the sentence
			training_set = "> " + sentence.translate(None, string.punctuation)
			tokens = nltk.word_tokenize(training_set.lower())
			# Form the unigrams here
			uni_grams.extend(tokens)
		# Close the open file pointer	
		f.close()
		# Form the dictionaries here
		self.uni_grams_dict = nltk.FreqDist(uni_grams)
		self.bi_grams_dict = nltk.FreqDist(nltk.bigrams(uni_grams))
		self.turing_dict = {0:len(self.bi_grams_dict)}
		# Build a separate dictonary for Good Turing Discount
		for key in self.bi_grams_dict.keys():
			bi_gram_count = self.bi_grams_dict.get(key)
			if ( bi_gram_count in self.turing_dict ) :
				value = self.turing_dict.get(bi_gram_count)
				self.turing_dict[bi_gram_count] = value+1 
			else :
				self.turing_dict[bi_gram_count] = 1
				
	def buildBiGram(self,title,option):
	
		print ("\n\n_________________________________"+title+"_________________________________\n\n")
		count = 1
		f = open(self.file_path,'r')
		for sentence in self.tokenizer.tokenize(f.read()):
			print ("----------------------------------Sentence:%d----------------------------------" %(count))
			word_list = []
			words = sentence.translate(None, string.punctuation)
			word_list.extend(nltk.word_tokenize(words.lower()))
			# Now word list has the required set of words. Loop through it
			print ("\t"+" \t|  ".join(word_list))
			for i in word_list :
				tmp_string = ''
				for j in word_list:
					key = (i,j)
					tmp_string = tmp_string + str(self.bi_gram_all(key,option)) + " \t|  "
				print (i +" "+tmp_string+"\n")
			if ( option != 0 ):	 
				prob_sentence = Decimal(self.bi_gram_all(('>',word_list[0]),option))
				# Loop to calculate the probability
				for i in range(1,len(word_list) - 1):
					prob_sentence *= Decimal(self.bi_gram_all((word_list[i],word_list[i+1]),option))
				print ("Total Probability of the Sentence-%d is :" %count)
				print (Decimal(prob_sentence))
			count += 1
		f.close()
		
	def getGTCount(self,key):
		
		gt_count = key
		if( key < 11 ): # Assuming that smoothing is not required for c > 10
			# New count = ( key + 1 ) * ( value @ key + 1 ) / (value @ key)
			next_key = key + 1
			gt_count =  ( ( Decimal(next_key) * Decimal(self.turing_dict[next_key]) ) / Decimal( self.turing_dict[key] ) )
		return (gt_count)

	def bi_gram_all(self,key,option):
		
		b = key[0]
		num = 0
		den = 1
		getcontext().prec = 4
		if ( option == 0 ):
			# Computes the bigram counts
			num = self.bi_grams_dict.get(key,0)
					
		elif ( option == 1 ):
			# Computes p(a|b) = p(a,b) / p(b) to the fifth decimal	
			num = Decimal(self.bi_grams_dict.get(key,0))
			den = Decimal(self.uni_grams_dict.get(b,0))
		
		elif ( option == 2 ): # Corresponds to add one smoothing
			# Computes p*(a|b) = (c+1)*(N/(N+V)) to the fifth decimal
			num = Decimal(self.bi_grams_dict.get(key,0) + 1 )
			den = Decimal(self.uni_grams_dict.get(b,0) + len(self.uni_grams_dict))
			
		elif ( option == 3 ): # Corresponds to Good turing discount
			num = self.getGTCount(self.bi_grams_dict.get(key,0))
			den = Decimal(self.uni_grams_dict.get(b,0))
			
		if ( den == 0 ):
			num = 0
			den = 1						
		return (Decimal(num/den))
		
def main():
	
	parser = argparse.ArgumentParser(description='Program to compute probabilities using Bi-grams')
	parser.add_argument('-t','--train', help='Training corpus file name',required=True)
	parser.add_argument('-f','--file', help='File for which probablity should be calculated',required=True)
	args = parser.parse_args()
	ngrams=Ngrams(args.train,args.file)
	ngrams.buildBiGram("Bigram model count",0)
	ngrams.buildBiGram("Bigram model without smoothing",1)
	ngrams.buildBiGram("Bigram model with add-one smoothing",2)
	ngrams.buildBiGram("Bigram model with Good-Turing discounting",3)
	return 0

if __name__ == '__main__':
	main()
