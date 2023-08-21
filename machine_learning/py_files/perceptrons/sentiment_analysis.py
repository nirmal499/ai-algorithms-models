import turicreate as tc

import sys
sys.path.append('..')
import libs.utils as utils

movies = tc.SFrame('./IMDB_Dataset.csv')

"""
The dataset has two columns, one with the review, as a string, and one with the sentiment, as 
positive or negative. First, we need to process the string, because each of the words needs to be a 
different feature. The Turi Create built-in function count_words in the text_analytics package is useful for this task, because it turns a sentence into a dictionary with the word counts. For 
example, the sentence "to be or not to be" is turned into the dictionary {'to':2, 'be':2, 'or':1, 'not':1}. 
We add a new column called words containing this dictionary as follows:
"""
movies['words'] = tc.text_analytics.count_words(movies['review'])
model = tc.logistic_classifier.create(movies, features=['words'], target='sentiment')

"""
Now that we’ve trained our model, we can look at the weights of the words, with the coefficients command.
The table we obtain has several columns, but the ones we care about are 
index and value, which show the words and their weights. The top five follow:
1. (intercept): 0.065
2. if: –0.018
3. viewing: 0.089
4. comfortable: 0.517
5. become: 0.106

The first one, called intercept, is the bias. Because the bias of the model is positive, the empty review is positive
This makes sense, because users who rate movies negatively tend 
to leave a review, whereas many users who rate movies positively don't leave any review. The 
other words are neutral, so their weights don't mean very much

6. wonderful: 1.043
7. horrible: –1.075
8. the: 0.0005
As we see, the weight of the word wonderful is positive, the weight of the word horrible is negative, 
and the weight of the word the is small. This makes sense: wonderful is a positive word, horrible
is a negative word, and the is a neutral word
"""
weights = model.coefficients
# print(weights)

"""
Let's find the most positive and negative reviews. For this, we use the model to 
make predictions for all the movies. These predictions will be stored in a new column called 
predictions, using the following command
"""
movies['predictions'] = model.predict(movies, output_type='probability')

# Most positive review
print(movies.sort('predictions')[-1]) # Default sorts in ascending

# Most negative review
print(movies.sort('predictions')[0])