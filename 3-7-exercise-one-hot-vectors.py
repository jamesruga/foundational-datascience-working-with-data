#!/usr/bin/env python
# coding: utf-8

# # Exercise: Titanic Dataset - One-Hot Vectors
# 
# In this unit, we'll build a model to predict who survived the Titanic disaster. We'll practice transforming data between numerical and categorical types, including use of one-hot vectors.
# 
# ## Data prepartion
# 
# First, we'll open and quickly clean our dataset, like we did in the last unit:
# 

# In[1]:


import pandas
get_ipython().system('wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/m0c_logistic_regression.py')
get_ipython().system('wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/titanic.csv')

# Open our dataset from file
dataset = pandas.read_csv("titanic.csv", index_col=False, sep=",", header=0)

# Fill missing cabin information with 'Unknown'
dataset["Cabin"].fillna("Unknown", inplace=True)

# Remove rows missing Age information
dataset.dropna(subset=["Age"], inplace=True)

# Remove the Name, PassengerId, and Ticket fields
# This is optional; it makes it easier to read our print-outs
dataset.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

dataset.head()


# ## About Our Model
# 
# We'll training a model type known as Logistic Regression, which will predict who survives the Titanic disaster.
# 
# For this exercise, you don't need to understand logistic regression. We placed the implementation outside this notebook in a method called `train_logistic_regression`. If you're curious, you can read this method in our GitHub repository.
# 
# `train_logistic_regression`:
# 
# 1. Accepts our data frame and a list of features to include in the model. 
# 2. Trains the model.
# 3. Returns a number that states how well the model performs as it predicts passenger survival. **Smaller numbers are better.**
# 
# ## Numerical Only
# 
# Let's create a model that uses only the numerical features.
# 
# First, we'll use `Pclass` here as an ordinal feature, rather than a one-hot encoded categorical feature.

# In[2]:


from m0c_logistic_regression import train_logistic_regression

features = ["Age", "Pclass", "SibSp", "Parch", "Fare"] 
loss_numerical_only = train_logistic_regression(dataset, features)

print(f"Numerical-Only, Log-Loss (cost): {loss_numerical_only}")


# We have our starting point. Let's see if categorical features will improve the model.
# 
# ## Binary Categorical Features
# 
# Categorical features with only two potential values can be encoded in a single column, as `0` or `1`.
# 
# We'll convert `Sex` values into `IsFemale` - a `0` for male and `1` for female - and include that in our model.

# In[3]:


# Swap male / female with numerical values
# We can do this because there are only two categories
dataset["IsFemale"] = dataset.Sex.replace({'male':0, 'female':1})

# Print out the first few rows of the dataset
print(dataset.head())

# Run and test the model, also using IsFemale this time
features = ["Age", "Pclass", "SibSp", "Parch", "Fare", "IsFemale"] 
loss_binary_categoricals = train_logistic_regression(dataset, features)

print(f"\nNumerical + Sex, Log-Loss (cost): {loss_binary_categoricals}")


# Our loss (error) decreased! This model performs better than the previous model.
# 
# ## One-Hot Encoding
# 
# Ticket class (`Pclass`) is an Ordinal feature. Its potential values (1, 2 & 3) have an order and they have equal spacing. However, this even spacing might be incorrect - in stories about the Titanic, the third-class passengers were treated much worse than those in 1st and 2nd class.
# 
# Let's convert `Pclass` into a categorical feature using one-hot encoding:

# In[4]:


# Get all possible categories for the "PClass" column
print(f"Possible values for PClass: {dataset['Pclass'].unique()}")

# Use Pandas to One-Hot encode the PClass category
dataset_with_one_hot = pandas.get_dummies(dataset, columns=["Pclass"], drop_first=False)

# Add back in the old Pclass column, for learning purposes
dataset_with_one_hot["Pclass"] = dataset.Pclass

# Print out the first few rows
dataset_with_one_hot.head()


# Note that `Pclass` converted into three values: `Pclass_1`, `Pclass_2` and `Pclass_3`.
# 
# Rows with `Pclass` values of 1 have a value in the `Pclass_1` column. We see the same pattern for values of 2 and 3.
# 
# Now, re-run the model, and treat `Pclass` values as a categorical values, rather than ordinal values.

# In[5]:


# Run and test the model, also using Pclass as a categorical feature this time
features = ["Age", "SibSp", "Parch", "Fare", "IsFemale",
            "Pclass_1", "Pclass_2", "Pclass_3"]

loss_pclass_categorical = train_logistic_regression(dataset_with_one_hot, features)

print(f"\nNumerical, Sex, Categorical Pclass, Log-Loss (cost): {loss_pclass_categorical}")


# This seems to have made things slightly worse!
# 
# Let's move on.
# 
# ## Including Cabin
# 
# Recall that many passengers had `Cabin` information. `Cabin` is a categorical feature and should be a good predictor of survival, because people in lower cabins probably had little time to escape during the sinking.
# 
# Let's encode cabin using one-hot vectors, and include it in a model. This time, there are so many cabins that we won't print them all out. To practice printing them out, feel free to edit the code as practice.

# In[6]:


# Use Pandas to One-Hot encode the Cabin and Pclass categories
dataset_with_one_hot = pandas.get_dummies(dataset, columns=["Pclass", "Cabin"], drop_first=False)

# Find cabin column names
cabin_column_names = list(c for c in dataset_with_one_hot.columns if c.startswith("Cabin_"))

# Print out how many cabins there were
print(len(cabin_column_names), "cabins found")

# Make a list of features
features = ["Age", "SibSp", "Parch", "Fare", "IsFemale",
            "Pclass_1", "Pclass_2", "Pclass_3"] + \
            cabin_column_names

# Run the model and print the result
loss_cabin_categorical = train_logistic_regression(dataset_with_one_hot, features)

print(f"\nNumerical, Sex, Categorical Pclass, Cabin, Log-Loss (cost): {loss_cabin_categorical}")


# That's our best result so far!
# 
# ## Improving Power
# 
# Including very large numbers of categorical classes - for example, 135 cabins - is often not the best way to train a model, because the model only has a few examples of each category class to learn from.
# 
# Sometimes, we can improve models if we simplify features. `Cabin` was probably useful because it indicated which Titanic deck people were probably situated in: those in lower decks would have had their quarters flooded first. 
# 
# It might become simpler to use deck information, instead of categorizing people into Cabins. 
# 
# Let's simplify what we have run, replacing the 135 `Cabin` categories with a simpler `Deck` category that has only 9 values: A - G, T, and U (Unknown)

# In[7]:


# We have cabin names, like A31, G45. The letter refers to the deck that
# the cabin was on. Extract just the deck and save it to a column. 
dataset["Deck"] = [c[0] for c in dataset.Cabin]

print("Decks: ", sorted(dataset.Deck.unique()))

# Create one-hot vectors for:
# Pclass - the class of ticket. (This could be treated as ordinal or categorical)
# Deck - the deck that the cabin was on
dataset_with_one_hot = pandas.get_dummies(dataset, columns=["Pclass", "Deck"], drop_first=False)

# Find the deck names
deck_of_cabin_column_names = list(c for c in dataset_with_one_hot.columns if c.startswith("Deck_"))
 
features = ["Age", "IsFemale", "SibSp", "Parch", "Fare", 
            "Pclass_1", "Pclass_2", "Pclass_3",
            "Deck_A", "Deck_B", "Deck_C", "Deck_D", 
            "Deck_E", "Deck_F", "Deck_G", "Deck_U", "Deck_T"]

loss_deck = train_logistic_regression(dataset_with_one_hot, features)

print(f"\nSimplifying Cabin Into Deck, Log-Loss (cost): {loss_deck}")


# ## Comparing Models
# 
# Let's compare the `loss` for these models:

# In[8]:


# Use a dataframe to create a comparison table of metrics
# Copy metrics from previous Unit

l =[["Numeric Features Only", loss_numerical_only],
    ["Adding Sex as Binary", loss_binary_categoricals],
    ["Treating Pclass as Categorical", loss_pclass_categorical],
    ["Using Cabin as Categorical", loss_cabin_categorical],
    ["Using Deck rather than Cabin", loss_deck]]

pandas.DataFrame(l, columns=["Dataset", "Log-Loss (Low is better)"])


# We can see that including categorical features can both improve and harm how well a model works. Often, experimentation is the best way to find the best model. 
# 
# ## Summary
# 
# In this unit you learned how to use One-Hot encoding to address categorical data.
# 
# We also explored how sometimes critical thinking about the problem at hand can improve a solution more than simply including all possible features in a model.
