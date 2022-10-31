#!/usr/bin/env python
# coding: utf-8

# In[73]:


# load data libraries
import numpy as np # linear algebra library
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile # to read zip files
from sklearn.model_selection import train_test_split
import sklearn


# data understanding libraries
import matplotlib.pyplot as plt # ploting library
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter


# data preparation
import re
from nltk.stem import PorterStemmer


# ADS Creation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[3]:


train=pd.read_csv("C:\\Users\\ErinySamoel\\Downloads\\train.tsv.zip","\t")
test=pd.read_csv("C:\\Users\\ErinySamoel\\Downloads\\test.tsv.zip","\t")


# ### data understanding

# In[4]:


train.head(100)


# ### split data to train and test

# In[66]:


train_data, test_data = train_test_split(train, test_size=0.4, random_state=1)
val_data, test_data = train_test_split(test, test_size=0.5, random_state=1)

train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)


# In[36]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


vectorizer = TfidfVectorizer(min_df=1)
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)


# In[76]:


id_train, X_train, y_train =  train_preprocessed['PhraseId'], train_preprocessed['Phrase_txt'], train_preprocessed['Sentiment']


# In[60]:


print("Train set size is ",len(train_data))


# In[7]:


#data understanding
train.info()


# #### Data preperation

# In[8]:


words = " ".join([row["Phrase"]for ind, row in train.iterrows()]).split()


# In[9]:


train['Phrase'].str.split()


# In[10]:


len(set(words))


# In[11]:


train["Phrase"].value_counts()


# In[12]:


pd.Series([s for s in words if "-" in s]).value_counts()


# In[13]:


pd.Series([s for s in words if any(char.isdigit() for char in s)]).value_counts()


# In[14]:


pd.Series([s for s in words if "`" in s]).value_counts()


# In[15]:


pd.Series([s for s in words if ";" in s]).unique()


# In[16]:


pd.Series([s for s in words if "?" in s]).unique()


# In[17]:


pd.Series([s for s in words if "!" in s]).unique()


# In[18]:


pd.Series([s for s in words if "=" in s]).unique()


# In[19]:


pd.Series([s for s in words if ":" in s]).unique()


# In[20]:


stopwords = set(["RRB,LRB"])
porter = PorterStemmer()
# lancaster=LancasterStemmer()

def ret_words(words):
    words = words.lower()
    words = words.replace('-', ' ')
    words = words.replace('--', ' ')
    words = words.replace(',', ' ')
    words = words.replace('`', ' ')
    words = words.replace('`', '')
    words = words.replace('\'', '')
    words = words.replace(':', '')
    words = words.replace('?', '')
    words = words.replace(';', '')

    text = []
    for word in words.split():
        if re.findall('[0-9]', word): continue
        if word in stopwords: continue
        if len(word) > 0: text.append(porter.stem(re.sub(r'[^\w\s]','',word)))
    return ' '.join(text)

def preprocess(df,flag):
     # Convert list of ingredients to string
    df['Phrase_txt'] = df["Phrase"].apply(ret_words)
    return df
 


# In[21]:


train_preprocessed = preprocess(train_data,0)
test_preprocessed = preprocess(test_data,1)


# In[22]:


train_preprocessed.sample(10)


# #### ADS creation with BOW

# In[53]:


# BoW
BoW = CountVectorizer()

BoW.fit(X_train)
Count_data = BoW.transform(X_train)

BoW_X_train = pd.DataFrame(Count_data.toarray(),columns=BoW.get_feature_names())

BoW_X_train


# In[29]:


X_train.head()


# In[33]:


BoW.fit(X_train.head())
Count_data = BoW.transform(X_train.head())
BoW_X_train = pd.DataFrame(Count_data.toarray(),columns=BoW.get_feature_names())
BoW_X_train


# In[34]:


# TFIDF
TFIDF = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.25, norm='l2', encoding='latin-1',                ngram_range=(1, 2), stop_words='english')

TFIDF.fit(X_train)
Count_data = TFIDF.transform(X_train)
TFIDF_X_train = pd.DataFrame(Count_data.toarray(),columns=TFIDF.get_feature_names())


TFIDF_X_train


# In[35]:


TFIDF = TfidfVectorizer()
TFIDF.fit(X_train.head(5))
Count_data = TFIDF.transform(X_train.head(5))
TFIDF_X_train = pd.DataFrame(Count_data.toarray(),columns=TFIDF.get_feature_names())


TFIDF_X_train


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
vectorizer = TfidfVectorizer(min_df=10)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(test_data)
train_x = vect.transform(X_train)
train_x.toarray()


# ### Evaluation and Model Selection
# and calculate accurecy with LogisticRegression

# In[54]:



model = sklearn.linear_model.LogisticRegression(random_state=0)
model.fit(BoW_X_train, y_train)


# In[75]:


pred = model.predict(BoW_X_train)


# In[77]:


print(accuracy_score(y_train, pred))

