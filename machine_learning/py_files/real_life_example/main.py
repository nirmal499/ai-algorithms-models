import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

"""
This command stores the dataset into a Pandas DataFrame called raw_data
Pandas adds an extra column numbering all the elements in 
the dataset. Because the dataset already comes with this numbering, we can set this index to be 
that column by specifying index_col="PassengerId". For this reason, we may see that in this 
dataset, the rows are indexed starting from 1 instead of starting from 0 as is more common in 
practice
CSV files have That it has 891 rows and 12 columns
"""
raw_data = pandas.read_csv('./titanic.csv')

# Examining the length of the dataset: len(raw_data)
# Examining the columns in the dataset: raw_data.columns
# Examining several columns together: raw_data[["Name", "Age"]]

"""
In order to find out how many values in each column are NaN, we use the is_na (or is_null) func-
tion. The is_na function returns a 1 if the entry is NaN, and a 0 otherwise. Therefore, if we sum 
over these values, we get the number of entries that are NaN in every column
> raw_data.isna().sum()

We will see that columns with missing data are Age, which is missing 177 values; Cabin, 
which is missing 687 values; and Embarked, which is missing 2 values
"""

"""
When a column is missing too many values, the corresponding feature may not be useful to our 
model. In this case, Cabin does not look like a good feature. Out of 891 rows, 687 don’t have a 
value. This feature should be removed

The axis parameter, which is 1 when we want to drop a column and 0 when we want to drop a row
"""
clean_data = raw_data.drop('Cabin', axis=1)

"""
The Age column is missing only 177 values out of 891, which is not that many.
We can fill them in with the average or the median of the other values.
We calculate the median, using the median function, and we obtain 28. Next, we use the fillna function, 
which fills in the missing values with the value we give it
"""
median_age = clean_data["Age"].median()
clean_data["Age"] = clean_data["Age"].fillna(median_age)

"""
The third column that is missing values is Embarked, which is missing two values
We can lump all the passengers with no value in the Embarked column into the 
same class. We can call this class U, for "Unknown."
"""
clean_data["Embarked"] = clean_data["Embarked"].fillna('U')

"""
When Pandas loads a dataset, it adds an index column that numbers each of the elements. We 
can ignore this column, but when we save the dataset, we must set the parameter index = None to 
avoid saving unnecessary index columns
"""
clean_data.to_csv('./clean_titanic_data.csv', index = None)

"""
If we have any columns with categorical data, we must turn them into numbers. The way to do this effectively 
using a technique called one-hot encoding.

One-hot encoding works in the following way: First, we look at how many classes the feature has and build as 
many new columns. For example, a column with two categories, female and male, would turn it into two 
columns, one for female and one for male. We can call these columns gender_male and gender_female for clarity.
Then, we look at each passenger. If the passenger is female, then the gender_female column will have a 1, 
and the gender_male column will have a 0. If the passenger is male, then we do the opposite.

The Pandas function get_dummies helps us with one-hot encoding. We use it to create some new columns, then 
we attach these columns to the dataset, and we must not forget to remove the original column, because 
that information is redundant
"""

# Create columns with the one-hot encoded columns
gender_columns = pandas.get_dummies(preprocessed_data['Sex'], prefix='Sex')
embarked_columns = pandas.get_dummies(preprocessed_data["Embarked"], prefix="Embarked")

# Concatenates the dataset with the newly created columns
preprocessed_data = pandas.concat([preprocessed_data, gender_columns], axis=1)
preprocessed_data = pandas.concat([preprocessed_data, embarked_columns], axis=1)

preprocessed_data = preprocessed_data.drop(['Sex', 'Embarked'], axis=1)

categorized_pclass_columns = pandas.get_dummies(preprocessed_data['Pclass'], prefix='Pclass')
preprocessed_data = pandas.concat([preprocessed_data, categorized_pclass_columns], axis=1)
preprocessed_data = preprocessed_data.drop(['Pclass'], axis=1)

# BINNING
"""
We need to give the model all the freedom to determine which ages determine whether a passenger is more 
or less likely to survive. What can we do? Many nonlinear models can deal with this, but we should still 
modify the Age column to something that gives the model more freedom to explore the data. A useful 
technique we can do is to bin the ages, namely, split them into several different buckets. For example, 
we can turn the age column into the following:

From 0 to 10 years old
From 11 to 20 years old
From 21 to 30 years old
From 31 to 40 years old
From 41 to 50 years old
From 51 to 60 years old
From 61 to 70 years old
From 71 to 80 years old
81 years old or older

"""
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
categorized_age = pandas.cut(preprocessed_data['Age'], bins)
preprocessed_data['Categorized_age'] = categorized_age
preprocessed_data = preprocessed_data.drop(["Age"], axis=1)

# FEATURE SELECTION
"""
we should also drop some other columns, because they are not necessary for our model, or even worse, 
they may completely ruin our model.
Let’s look at the name feature. Should we consider it in our model? Absolutely not, and here 
is the reason: every passenger has a different name (perhaps with some very few exceptions, 
which are not significant). Therefore, the model would be trained to simply learn the names of 
the passengers who survived, and it wouldn’t be able to tell us anything about new passengers 
whose names it hasn’t seen
"""
preprocessed_data = preprocessed_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1)

preprocessed_data.to_csv('preprocessed_titanic_data.csv', index=None)

data = pandas.read_csv('./preprocessed_titanic_data.csv')

"""
What about the Survived feature—shouldn’t we get rid of that one, too? Definitely! Keeping the Survived column in 
our dataset while training will overfit, because the model will simply use this feature to determine whether 
the passenger survived. This is like cheating on a test by looking at the solution. We won’t remove it yet from 
the dataset, because we will remove it when we split the dataset into features and labels later for training
"""
features = data.drop(["Survived"], axis=1)
labels = data["Survived"]

# FEATURES-LABELS SPLIT AND TRAIN-VLIDATION SPLIT
"""
We split the data into training and validation sets. We’ll use 60% of our data for training, 20% for validation, 
and 20% for testing. In this function, we specify the percentage of data we want for validation with the 
test_size parameter. The output is the four tables called features_train, features_test, labels_train, labels_test

We want 60% training, 20% validation, and 20% testing, we need to use the train_test_split function twice: 
once for separating the training data, and once for splitting the validation and testing sets, as shown here:
"""

# remark: we fix random_state the end, to make sure we always get the same split
features_train, features_validation_test, labels_train, labels_validation_test = 
	train_test_split(features, labels, test_size=0.4, random_state=100)

features_validation, features_test, labels_validation, labels_test = 
	train_test_split(features_validation_test, labels_validation_test, test_size=0.5, random_state=100)

# print(len(features_train))
# print(len(features_validation))
# print(len(features_test))
# print(len(labels_train))
# print(len(labels_validation))
# print(len(labels_test))

# Training different models on our dataset

lr_model = LogisticRegression()
lr_model.fit(features_train, labels_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(features_train, labels_train)

nb_model = GaussianNB()
nb_model.fit(features_train, labels_train)

svm_model = SVC()
svm_model.fit(features_train, labels_train)

rf_model = RandomForestClassifier()
rf_model.fit(features_train, labels_train)

gb_model = GradientBoostingClassifier()
gb_model.fit(features_train, labels_train)

ab_model = AdaBoostClassifier()
ab_model.fit(features_train, labels_train)

# EVALUATING THE MODELS
# Accuracy
print("Scores of the models")
print("Logistic regression:", lr_model.score(features_validation, labels_validation))
print("Decision tree:", dt_model.score(features_validation, labels_validation))
print("Naive Bayes:", nb_model.score(features_validation, labels_validation))
print("SVM:", svm_model.score(features_validation, labels_validation))
print("Random forest:", rf_model.score(features_validation, labels_validation))
print("Gradient boosting:", gb_model.score(features_validation, labels_validation))
print("AdaBoost:", ab_model.score(features_validation, labels_validation))

# F1-score
print("F1-scores of the models:")

lr_predicted_labels = lr_model.predict(features_validation)
print("Logistic regression:", f1_score(labels_validation, lr_predicted_labels))

dt_predicted_labels = dt_model.predict(features_validation)
print("Decision Tree:", f1_score(labels_validation, dt_predicted_labels))

nb_predicted_labels = nb_model.predict(features_validation)
print("Naive Bayes:", f1_score(labels_validation, nb_predicted_labels))

svm_predicted_labels = svm_model.predict(features_validation)
print("Support Vector Machine:", f1_score(labels_validation, svm_predicted_labels))

rf_predicted_labels = rf_model.predict(features_validation)
print("Random Forest:", f1_score(labels_validation, rf_predicted_labels))

gb_predicted_labels = gb_model.predict(features_validation)
print("Gradient boosting:", f1_score(labels_validation, gb_predicted_labels))

ab_predicted_labels = ab_model.predict(features_validation)
print("AdaBoost:", f1_score(labels_validation, ab_predicted_labels))

# TESTING THE MODEL
# Finding the accuracy and the F1-score of the model in the testing set.
gb_model.score(features_test, labels_test)

gb_predicted_test_labels = gb_model.predict(features_test)
# print(f1_score(labels_test, gb_predicted_test_labels))

"""
we trained these models without touching their hyperparameters, which means that Scikit-Learn picked some 
standard hyperparameters for them. we see a useful technique to 
search among many combinations of hyperparameters to find a good model for our data.
The performance of the gradient boosted tree was about as high as one can obtain for the 
Titanic dataset, so let’s leave that one alone. The poor SVM, however, performed last, with an accuracy of 
69% and an F1-score of 0.42. We believe in SVMs, however, because they are a powerful machine learning model. 
Perhaps the bad performance of this SVM is due to the hyperparameters it is using. 
There may be a better combination of them that works
"""

"""
Let’s try to train an SVM with two values for gamma, namely, 1 and 10. Why 1 and 10? Normally when we search for 
hyperparameters, we tend to do an exponential search, so we would try values such as 0.1, 1, 10, 
100, 1000, and so on, as opposed to 1, 2, 3, 4, 5. This exponential search covers a larger space and 
gives us better chances of finding good hyperparameters,
"""
print("SVM grid search with a radial basis function kernel")

# rbf, C=1, gamma=0.1
svm_1_01 = SVC(kernel='rbf', C=1, gamma=0.1)
svm_1_01.fit(features_train, labels_train)
print("C=1, gamma=0.1", svm_1_01.score(features_validation, labels_validation))

# rbf, C=1, gamma=1
svm_1_1 = SVC(kernel='rbf', C=1, gamma=1)
svm_1_1.fit(features_train, labels_train)
print("C=1, gamma=1", svm_1_1.score(features_validation, labels_validation))

# rbf, C=1, gamma=10
svm_1_10 = SVC(kernel='rbf', C=1, gamma=10)
svm_1_10.fit(features_train, labels_train)
print("C=1, gamma=10", svm_1_10.score(features_validation, labels_validation))

# rbf, C=10, gamma=0.1
svm_10_01 = SVC(kernel='rbf', C=10, gamma=0.1)
svm_10_01.fit(features_train, labels_train)
print("C=10, gamma=0.1", svm_10_01.score(features_validation, labels_validation))

# rbf, C=10, gamma=1
svm_10_1 = SVC(kernel='rbf', C=10, gamma=1)
svm_10_1.fit(features_train, labels_train)
print("C=10, gamma=1", svm_10_1.score(features_validation, labels_validation))

# rbf, C=10, gamma=10
svm_10_10 = SVC(kernel='rbf', C=10, gamma=10)
svm_10_10.fit(features_train, labels_train)
print("C=10, gamma=10", svm_10_10.score(features_validation, labels_validation))

"""
Now we evaluate them, using accuracy (another arbitrary choice—we could also use F1-score, 
precision, or recall). Notice that the best model among these is the one with gamma = 
0.1 and C = 10, with an accuracy of 0.72.
The best accuracy is 0.72, given by the model with gamma = 0.1 and 
C = 1. This is an improvement from the 0.68 we obtained previously when we didn’t specify any 
hyperparameters.
"""

# GRID SEARCH
"""
Using the GridSearchCV object. First, we define the hyperparameters as a dictionary, where the key of 
the dictionary is the name of the parameter and the value corresponding to this key is the list of 
values we want to try for our hyperparameter. In this case, let’s explore the following combinations of 
hyperparameters

Kernel: RBF 
C: 0.01, 0.1, 1, 10, 100
gamma: 0.01, 0.1, 1, 10, 100
"""

# A dictionary with the hyperparameters and the values we want to try
svm_parameters = {
					'kernel': ['rbf'],
                  	'C': [0.01, 0.1, 1 , 10, 100],
                  	'gamma': [0.01, 0.1, 1, 10, 100]
}

# A regular SVM with no hyperparameters
svm = SVC()

# A GridSearchCV object where we pass the SVM and the hyperparameter dictionary
svm_gs = GridSearchCV(estimator = svm, param_grid = svm_parameters)

# We fit the GridSearchCV model in the same way that we fit a regular model in Scikit-Learn
svm_gs.fit(features_train, labels_train)

# This trains 25 models with all the combinations of hyperparameters given in the hyperparameter dictionary. 
# Now, we pick the best of these models and call it svm_winner
svm_winner = svm_gs.best_estimator_svm_winner
svm_winner.score(features_validation, labels_validation)

# The winning model used an RBF kernel with gamma = 0.01 and C = 10.

"""
CV at the end of GridSearchCV stands for cross-validation.
"""
# print(svm_gs.cv_results_)





