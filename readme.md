### Instructions to build the code:

# requirements:
python3
required python packages - sklearn, nltk, pandas, numpy

# To train models:
1. cd sentiment_predictor_assignemnt
2. in file sentiment_predictor.py, set self.svm = 1 if want to train svm, set self.svm = 0 if want to train Logistic regression
3. python3 sentiment_predictor.py

### models Evaluation results

1. Logistic regression
features used - TF-IDF
train accuracy - 94.58 %
test accuracy - 83.5 %
F1 score - 83.58 %

2. SVM
features used - TF-IDF
kernel - linear
train accuracy - 96.41 %
test accuracy - 84.33 %
F1 score - 84.59 %

performance --- both of these models performs equally with almost same evaluation accuracy and f1 score.

pros:
1. performs well on the text data even when the number of features are greater than the number of examples

cons:
1. does not take into account the sequence in the text.
2. Logistic regression is fast to train compared to SVM

### To test:

1. run the flask server with this command - python3 test_api.py

2. Test sentiment predictor from browser:

	paste this url in your browser: http://127.0.0.1:5000/
	and enter the appropriate inputs - "text" and "model name"

### tested exaples:
# correct
it is not a good phone, returned it on next day - negative
it is not a good phone, its an excellent phone - positive
i think nobody is going to buy this product - negative
it didnt meet my expectations - Positive

#wrong
the 2nd half could have been better - poistive
it was an action packed movie with full on entertainment - negative
