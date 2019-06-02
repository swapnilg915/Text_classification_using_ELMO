### https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb

import re, os, traceback, json, unidecode, string, pickle
from unicodedata import normalize
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from pandas import ExcelWriter
import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from many_stop_words import get_stop_words
from collections import defaultdict

from sklearn.model_selection import train_test_split, ShuffleSplit

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Input, Flatten, concatenate, Activation
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalMaxPool1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.engine import Layer
import keras.layers as layers

import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K

# Initialize session
sess = tf.Session()
K.set_session(sess)

# import spacy
# nlp= spacy.load('en_core_web_md', disable=['parser','ner'])

class ElmoEmbeddingLayer(Layer):
	def __init__(self, **kwargs):
		self.dimensions = 1024
		self.trainable=True
		super(ElmoEmbeddingLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
							   name="{}_module".format(self.name))

		self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
		super(ElmoEmbeddingLayer, self).build(input_shape)

	def call(self, x, mask=None):
		result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
					  as_dict=True,
					  signature='default',
					  )['default']
		return result

	def compute_mask(self, inputs, mask=None):
		return K.not_equal(inputs, '--PAD--')

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.dimensions)


class TwitterSentimentAnalysis(object):
	
	def __init__(self):

		self.embedding_dim = 1024
		
		# configurable parameters
		self.max_review_length = 40
		self.top_words = 100
		self.test_size = 0.1
		self.lemma_flag = 0
		self.max_letters = 2
		self.threshold = 0.5
		self.remove_stopwords = 1
		self.batch_size = 128
		self.epochs = 1

		self.stopwords = stopwords.words('english')
		if self.remove_stopwords: 
			# self.stopWords = list(get_stop_words('en'))
			# self.stopwords = []
			print("\n len(self.stopwords) = ", len(self.stopwords))
		self.tokenizer = Tokenizer(num_words=self.top_words)
		self.normalize_mapping = json.load(open("data/normalize_mapping.json"))


	def clean_text(self, text): # unused
		try:
			text = self.normalizeMapping(text)
			# text = re.sub(r"[^A-Za-z0-9@#$]", " ", text)
			# text = re.sub(r"[^A-Za-z0-9]", " ", text)
			text = re.sub(r"[^A-Za-z]", " ", text)
			text = self.removeStopWords(text)
			text = self.getLemma(text)
			# text = self.getStemming(text)
			text = re.sub(r"\s+", " ", text)
			text = " ".join([word for word in word_tokenize(text) if len(word) > 2])
			text = self.handle_unicode(text)
			text = re.sub(r"\s+", " ", text)
		except Exception as e:
			print("\n Error in clean_text --- ",e)
			print("\n text with error --- ",text, type(text))
		return text.lower().strip()

	def handle_unicode(self, val):
		return unidecode.unidecode(val)
	
	def readData(self):
		try:
			data  = pd.read_csv('data/text_classification_dataset.csv')
			print('\n train size == ', len(data))
			data['reviews'] = data['reviews'].apply(self.clean_text)
			print("\n data['reviews'] --- ", len(data['reviews']), len(data['labels']))
			return list(data['reviews']), list(data['labels'])

			# Load data from files
		    # positive_examples = list(open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())
		    # positive_examples = [s.strip() for s in positive_examples]
		    # negative_examples = list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
		    # negative_examples = [s.strip() for s in negative_examples]
		    # # Split by words
		    # x_text = positive_examples + negative_examples
		    # x_text = [clean_str(sent) for sent in x_text]
		    # x_text = [s.split(" ") for s in x_text]
		    # # Generate labels
		    # positive_labels = [[0, 1] for _ in positive_examples]
		    # negative_labels = [[1, 0] for _ in negative_examples]
		    # y = np.concatenate([positive_labels, negative_labels], 0)
		    # print("\n x_text --- ", x_text, "\n\n ", y)
		    # return x, y

		 #    data = pd.read_csv('jd_vs_not_feb19.csv')
		 #    print('\n train size == ', len(data))
			# data['plain_body'] = data['plain_body'].apply(self.clean_text)
			# print("\n data['plain_body'] --- ", len(data['plain_body']), len(data['Job Des.']))
			# examples = list(data['plain_body'])
			# labels = list(data['Job Des.'])
			# return examples, labels

		except Exception as e:
			print("\n Error in readData --- ", e,"\n",traceback.format_exc())
		# return data['reveiws'], data['labels']
		
	def splitData(self, examples, labels):
		x_tr, x_te, y_tr, y_te = train_test_split(examples, labels, test_size = 0.2, random_state=42)
		print("\n training size --- ", len(x_tr), len(y_tr))
		print("\n testing size --- ", len(x_te), len(y_te))
		return x_tr, x_te, y_tr, y_te

	def normalizeMapping(self, query):
		try:
			splited_query = query.split()
			for index, word in enumerate(splited_query):
				word = word.lower()
				if word in self.normalize_mapping:
					splited_query[index] = self.normalize_mapping[word]
			query = " ".join(splited_query)
			return query
		except Exception as e:
			print("\n Error in normalizeMapping --- ",traceback.format_exc())
			return query

	def clean_str(self, text):
		text = self.normalizeMapping(text)
		text = re.sub(r"[^a-zA-Z0-9.?!,:\n$]", " ", text)
		text = re.sub(r"\s{2,}", " ", text)
		text = re.sub(r"\s+", " ", text)
		return text.strip().lower()

	def removeStopWords(self, text):
		return " ".join([token for token in text.split() if token not in self.stopwords])
			
	def checkLemma(self, wrd):
		return nltk.stem.WordNetLemmatizer().lemmatize(nltk.stem.WordNetLemmatizer().lemmatize(wrd, 'v'), 'n')

	def getLemma(self, text):
		return " ".join([self.checkLemma(tok) for tok in text.lower().split()])

	def dataPreprocessing(self, examples):
		try:
			if self.lemma_flag: examples = [ self.getLemma(sent) for sent in examples]
			examples = [" ".join([word for word in word_tokenize(sent) if len(word)>self.max_letters and word not in self.stopwords]) for sent in examples ]
		except Exception as e:
			print("\n ERROR in dataPreprocessing --- ",e,"\n",traceback.format_exc())
		return examples


	# Function to build model
	def build_model(self): 
		try:
			input_text = Input(shape=(1,), dtype="string")
			embedding = ElmoEmbeddingLayer()(input_text)
			dense = layers.Dense(256, activation='relu')(embedding)
			pred = layers.Dense(1, activation='sigmoid')(dense)

			model = Model(inputs=[input_text], outputs=pred)

			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
			model.summary()
		except Exception as e:
			print("\n Error in build_model --- ", e,"\n ",traceback.format_exc())
		return model


	def save_models(self, model):
		try:
			model.save('sentiment_predictor_12.h5')
			with open('tokenizer_sentiment_predictor_12.pkl', 'wb') as f:
				pickle.dump(self.tokenizer, f)
		except Exception as e:
			print ("\n error in model saving = ",e,"\n ",traceback.format_exc())

	def test(self,x_te, model):
		sequences = self.tokenizer.texts_to_sequences(x_te)
		x_predict = sequence.pad_sequences(sequences, maxlen=self.max_review_length)
		y_prob = model.predict(x_predict)
		
		test_results = [lst[0] for lst in y_prob]
		test_labels = [0 if score < self.threshold else 1 for score in test_results ]
		
		print("\n len(test_labels) === ",len(test_labels))
		return test_labels


	def main(self):
		examples, labels = self.readData()
		data['reviews'] = self.dataPreprocessing(examples)

		x_tr, x_te, y_tr, y_te = self.splitData(examples, labels)

		model = self.build_model()
		model.fit(x_tr, y_tr, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size)
		test_labels = self.test(x_te, model)

		
if __name__ == '__main__':
	obj = TwitterSentimentAnalysis()
	obj.main()
	
