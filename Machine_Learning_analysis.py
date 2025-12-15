#!/usr/bin/env python
# coding: utf-8

# In[99]:


# Standard library
import os
import re
import string
import html
import warnings
from collections import Counter

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import nltk

# NLTK
from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import word_tokenize, TweetTokenizer

# Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier

# Transformers
from transformers import AutoTokenizer

# NLTK downloads
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)

# Pandas display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# Warnings and environment
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[100]:


PYDEVD_DISABLE_FILE_VALIDATION=1


# In[ ]:





# In[101]:


# load datasets

#======
#change to dataset file path
file_path = "/Users/hollytaswell/Desktop/Raw_data/"
#======

#load files
ds_fem_file = file_path+"DS_female.csv"
ds_male_file = file_path+"DS_male.csv"

rec_fem_file = file_path+"rec_female.csv"
rec_male_file = file_path+"rec_male.csv"

card_fem_file = file_path+"card_female.csv"
card_male_file = file_path+"card_male.csv"


# In[102]:


#load ds
ds_fem_df = pd.read_csv(ds_fem_file)
ds_f_df = ds_fem_df[['response']]
ds_f_df = ds_f_df.copy()
ds_f_df['gender'] = 1

ds_male_df = pd.read_csv(ds_male_file)
ds_m_df = ds_male_df[['response']]
ds_m_df = ds_m_df.copy()
ds_m_df['gender'] = 0

#load rec 
rec_fem_df = pd.read_csv(rec_fem_file)
rec_f_df = rec_fem_df[['response']]
rec_f_df = rec_f_df.copy()
rec_f_df['gender'] = 1

rec_male_df = pd.read_csv(rec_male_file)
rec_m_df = rec_male_df[['response']]
rec_m_df = rec_m_df.copy()
rec_m_df['gender'] = 0

#load card
card_fem_df = pd.read_csv(card_fem_file)
card_f_df = card_fem_df[['response']]
card_f_df = card_f_df.copy()
card_f_df['gender'] = 1

card_male_df = pd.read_csv(card_male_file)
card_m_df = card_male_df[['response']]
card_m_df = card_m_df.copy()
card_m_df['gender'] = 0


# In[103]:


#combine male and female cover letters into datasets
df = pd.concat([ds_f_df, ds_m_df,rec_f_df, rec_m_df, card_f_df, card_m_df])
ds_df = pd.concat([ds_f_df, ds_m_df])
rec_df = pd.concat([rec_f_df, rec_m_df])
card_df = pd.concat([card_f_df, card_m_df])


# In[104]:


dataset = df


# In[105]:


#data cleaning and standardization
df['response'] = (
    df['response']
        # normalize whitespace and casing
        .str.replace('\n\n', ' ', regex=False)
        .str.replace('\n', ' ', regex=False)
        .str.lower()
        .str.strip()

        # remove placeholders like [NAME], [COMPANY]
        .str.replace(r"\[.*?\]", "", regex=True)

        # punctuation cleanup
        .str.replace(",", "", regex=False)
        .str.replace("’", "", regex=False)
        .str.replace("'", "", regex=False)

        # remove synthetic company / location artifacts
        .str.replace(r"\bgoogle\w*\b", "", regex=True)
        .str.replace(r"\bblue banking\w*\b", "", regex=True)
        .str.replace(r"\bsunny offices\w*\b", "", regex=True)
        .str.replace(r"\bblue office\w*\b", "", regex=True)
        .str.replace(r"\bsunshine general hospital\w*\b", "", regex=True)
        .str.replace(r"\bsunshine\w*\b", "", regex=True)
        .str.replace(r"\bblue day general\w*\b", "", regex=True)

        # remove role related boilerplate
        .str.replace(r"\bhiring manager\w*\b", "", regex=True)
        .str.replace(r"\bsincerely\w*\b", "", regex=True)
        .str.replace(r"\bdear\w*\b", "", regex=True)
        .str.replace(r"\bhiring\w*\b", "", regex=True)

        # remove profession specific terms
        .str.replace(r"\bdata\w*\b", "", regex=True)
        .str.replace(r"\bscient\w*\b", "", regex=True)
        .str.replace(r"\bdata science\w*\b", "", regex=True)
        .str.replace(r"\bcardi\w*\b", "", regex=True)
        .str.replace(r"\brecept\w*\b", "", regex=True)

        # final whitespace cleanup
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
)



ds_df['response'] = (
    ds_df['response']
        # normalize whitespace and casing
        .str.replace('\n\n', ' ', regex=False)
        .str.replace('\n', ' ', regex=False)
        .str.lower()
        .str.strip()

        # remove placeholders like [NAME], [COMPANY]
        .str.replace(r"\[.*?\]", "", regex=True)

        # punctuation cleanup
        .str.replace(",", "", regex=False)
        .str.replace("’", "", regex=False)
        .str.replace("'", "", regex=False)

        # remove synthetic company / location artifacts
        .str.replace(r"\bgoogle\w*\b", "", regex=True)
        .str.replace(r"\bblue banking\w*\b", "", regex=True)
        .str.replace(r"\bsunny offices\w*\b", "", regex=True)
        .str.replace(r"\bblue office\w*\b", "", regex=True)
        .str.replace(r"\bsunshine general hospital\w*\b", "", regex=True)
        .str.replace(r"\bsunshine\w*\b", "", regex=True)
        .str.replace(r"\bblue day general\w*\b", "", regex=True)

        # remove role related boilerplate
        .str.replace(r"\bhiring manager\w*\b", "", regex=True)
        .str.replace(r"\bsincerely\w*\b", "", regex=True)
        .str.replace(r"\bdear\w*\b", "", regex=True)
        .str.replace(r"\bhiring\w*\b", "", regex=True)

        # remove profession specific terms
        .str.replace(r"\bdata\w*\b", "", regex=True)
        .str.replace(r"\bscient\w*\b", "", regex=True)
        .str.replace(r"\bdata science\w*\b", "", regex=True)
        .str.replace(r"\bcardi\w*\b", "", regex=True)
        .str.replace(r"\brecept\w*\b", "", regex=True)

        # final whitespace cleanup
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
)


rec_df['response'] = (
    rec_df['response']
        # normalize whitespace and casing
        .str.replace('\n\n', ' ', regex=False)
        .str.replace('\n', ' ', regex=False)
        .str.lower()
        .str.strip()

        # remove placeholders like [NAME], [COMPANY]
        .str.replace(r"\[.*?\]", "", regex=True)

        # punctuation cleanup
        .str.replace(",", "", regex=False)
        .str.replace("’", "", regex=False)
        .str.replace("'", "", regex=False)

        # remove synthetic company / location artifacts
        .str.replace(r"\bgoogle\w*\b", "", regex=True)
        .str.replace(r"\bblue banking\w*\b", "", regex=True)
        .str.replace(r"\bsunny offices\w*\b", "", regex=True)
        .str.replace(r"\bblue office\w*\b", "", regex=True)
        .str.replace(r"\bsunshine general hospital\w*\b", "", regex=True)
        .str.replace(r"\bsunshine\w*\b", "", regex=True)
        .str.replace(r"\bblue day general\w*\b", "", regex=True)

        # remove role related boilerplate
        .str.replace(r"\bhiring manager\w*\b", "", regex=True)
        .str.replace(r"\bsincerely\w*\b", "", regex=True)
        .str.replace(r"\bdear\w*\b", "", regex=True)
        .str.replace(r"\bhiring\w*\b", "", regex=True)

        # remove profession specific terms
        .str.replace(r"\bdata\w*\b", "", regex=True)
        .str.replace(r"\bscient\w*\b", "", regex=True)
        .str.replace(r"\bdata science\w*\b", "", regex=True)
        .str.replace(r"\bcardi\w*\b", "", regex=True)
        .str.replace(r"\brecept\w*\b", "", regex=True)

        # final whitespace cleanup
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
)


card_df['response'] = (
    card_df['response']
        # normalize whitespace and casing
        .str.replace('\n\n', ' ', regex=False)
        .str.replace('\n', ' ', regex=False)
        .str.lower()
        .str.strip()

        # remove placeholders like [NAME], [COMPANY]
        .str.replace(r"\[.*?\]", "", regex=True)

        # punctuation cleanup
        .str.replace(",", "", regex=False)
        .str.replace("’", "", regex=False)
        .str.replace("'", "", regex=False)

        # remove synthetic company / location artifacts
        .str.replace(r"\bgoogle\w*\b", "", regex=True)
        .str.replace(r"\bblue banking\w*\b", "", regex=True)
        .str.replace(r"\bsunny offices\w*\b", "", regex=True)
        .str.replace(r"\bblue office\w*\b", "", regex=True)
        .str.replace(r"\bsunshine general hospital\w*\b", "", regex=True)
        .str.replace(r"\bsunshine\w*\b", "", regex=True)
        .str.replace(r"\bblue day general\w*\b", "", regex=True)

        # remove role related boilerplate
        .str.replace(r"\bhiring manager\w*\b", "", regex=True)
        .str.replace(r"\bsincerely\w*\b", "", regex=True)
        .str.replace(r"\bdear\w*\b", "", regex=True)
        .str.replace(r"\bhiring\w*\b", "", regex=True)

        # remove profession specific terms
        .str.replace(r"\bdata\w*\b", "", regex=True)
        .str.replace(r"\bscient\w*\b", "", regex=True)
        .str.replace(r"\bdata science\w*\b", "", regex=True)
        .str.replace(r"\bcardi\w*\b", "", regex=True)
        .str.replace(r"\brecept\w*\b", "", regex=True)

        # final whitespace cleanup
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
)


# In[106]:


#create colummn without stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    filtered = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered)

df['response_no_sw'] = df['response'].apply(remove_stopwords)
ds_df['response_no_sw'] = ds_df['response'].apply(remove_stopwords)
rec_df['response_no_sw'] = rec_df['response'].apply(remove_stopwords)
card_df['response_no_sw'] = card_df['response'].apply(remove_stopwords)


# In[107]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Get the cleaned text per group
fem_texts = df[df['gender'] == 1]['response_no_sw']
male_texts = df[df['gender'] == 0]['response_no_sw']

# Combine into one long text each
fem_text_combined = " ".join(fem_texts)
male_text_combined = " ".join(male_texts)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Female word cloud
wc_fem = WordCloud(width=800, height=400, background_color="white", colormap ="gist_heat", collocations = False).generate(fem_text_combined)
ax1.imshow(wc_fem, interpolation="bilinear")
ax1.set_title("Female Wordcloud", fontsize=16)
ax1.axis("off")

# Male word cloud
wc_male = WordCloud(width=800, height=400, background_color="white", colormap="gist_heat",collocations = False).generate(male_text_combined)
ax2.imshow(wc_male, interpolation="bilinear")
ax2.set_title("Male Wordcloud", fontsize=16)
ax2.axis("off")

plt.tight_layout()
plt.show()


# In[108]:


#Calculate Jaccard Similarity
#separate by gender

female_responses = dataset[dataset['gender']==1]['response_no_sw']
male_responses = dataset[dataset['gender']==0]['response_no_sw']

# Convert to sets of unique words
female_words = set(' '.join(female_responses).split())
male_words = set(' '.join(male_responses).split())

# Calculate Jaccard similarity
intersection = len(female_words.intersection(male_words))
union = len(female_words.union(male_words))
jaccard_similarity = intersection / union

print("==================================")
print(f"Jaccard similarity between female and male tweet vocabularies: {jaccard_similarity:.4f}")
print(f"Shared words: {intersection}")
print(f"Unique words total: {union}")
print("==================================")


# In[109]:


def simple_tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    return tokens

dataset["tokens"] = dataset["response_no_sw"].apply(simple_tokenize)

def type_token_ratio(tokens):
    if not tokens:
        return 0
    return len(set(tokens)) / len(tokens)

def hapax_rate(tokens):
    c = Counter(tokens)
    hapax = [w for w, n in c.items() if n == 1]
    return len(hapax) / len(tokens) if tokens else 0

dataset["ttr"] = dataset["tokens"].apply(type_token_ratio)
dataset["hapax_rate"] = dataset["tokens"].apply(hapax_rate)

print("============================")
print("Type Token Ratio and Hapax Rate")
print(dataset.groupby("gender")[["ttr", "hapax_rate"]].describe())
print("============================")


# In[110]:


def sentence_lengths(text):
    sentences = re.split(r"[.!?]+", text)
    lengths = []
    for s in sentences:
        tokens = simple_tokenize(s)
        if tokens:
            lengths.append(len(tokens))
    return lengths

dataset["sent_lengths"] = dataset["response"].apply(sentence_lengths)
dataset["avg_sent_len"] = dataset["sent_lengths"].apply(lambda L: np.mean(L) if L else 0)

def word_lengths(tokens):
    return [len(t) for t in tokens]

dataset["word_lengths"] = dataset["tokens"].apply(word_lengths)
dataset["avg_word_len"] = dataset["word_lengths"].apply(lambda L: np.mean(L) if L else 0)

print("============================")
print("Sentence and Word Length")
print(dataset.groupby("gender")[["avg_sent_len", "avg_word_len"]].describe())
print("============================")


# In[111]:


print("\n")
print("============================")
print("PREDICTIVE ANALYSIS")
print("============================")


# In[112]:


# first split off the test set (15 percent)
train_val, test = train_test_split(
    df,
    test_size=0.2,
    stratify=df['gender'],
    random_state=42
)

# then split train and val so that val is 15 percent of the full data
# val needs to be 15/85 of the remaining data
val_ratio = 0.2 / 0.85

train, val = train_test_split(
    train_val,
    test_size=val_ratio,
    stratify=train_val['gender'],
    random_state=42
)

print(train.shape, val.shape, test.shape)


# In[113]:


# lOGISTIC REGRESSION WITH STOP WORDS
#including stop words as produced a higher F1 score
X_train = train['response']       
y_train = train['gender']     

X_val = val['response']
y_val = val['gender']

X_test = test['response']       
y_test = test['gender']    

print("============================")
print("LOGISTIC REGRESSION WITH STOP WORDS")


# In[114]:


X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True)),
    ("clf", LogisticRegression(max_iter=2000))
], memory="cache_dir")

# joint hyperparameter grid using best selection
param_grid = {
    # TF-IDF params
    "tfidf__ngram_range": [(1, 2)],
    "tfidf__min_df": [1],
    "tfidf__max_df": [0.7],

    # Logistic Regression params
    "clf__C": [1],
    "clf__penalty": ["l2"],
    "clf__solver": ["liblinear"]    
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring="f1_macro",
    refit = True 
)

# fit on train only
grid.fit(X_train_full, y_train_full)

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

# evaluate best pipeline on val
best_pipe = grid.best_estimator_

# refit on train + val before final test evaluation
best_pipe.fit(X_train_full, y_train_full)

y_test_pred = best_pipe.predict(X_test)
print("Test weighted F1:", f1_score(y_test, y_test_pred, average="weighted"))
print("Test accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# feature importance from the fitted pipeline
tfidf = best_pipe.named_steps["tfidf"]
clf = best_pipe.named_steps["clf"]

feature_names = tfidf.get_feature_names_out()
coefs = clf.coef_[0]   # binary case

top_n = 10
top_female_idx = coefs.argsort()[::-1][:top_n]
top_male_idx = coefs.argsort()[:top_n]

print("\nTop female associated features:")
for idx in top_female_idx:
    print(feature_names[idx], coefs[idx])

print("\nTop male associated features:")
for idx in top_male_idx:
    print(feature_names[idx], coefs[idx])

print("==============================")


# In[115]:


#LOGISTIC REGRESSION WITH STOP WORDS
X_train_nosw = train['response_no_sw']       
y_train = train['gender']     

X_val_nosw = val['response_no_sw']
y_val = val['gender']

X_test_nosw = test['response_no_sw']       
y_test = test['gender']

print("============================")
print("LOGISTIC REGRESSION WITHOUT STOP WORDS")


# In[116]:


X_train_full = pd.concat([X_train_nosw, X_val_nosw])
y_train_full = pd.concat([y_train, y_val])

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True)),
    ("clf", LogisticRegression(max_iter=2000))
], memory="cache_dir")

# joint hyperparameter grid
param_grid = {
    # TF-IDF params
    "tfidf__ngram_range": [(1,1)],
    "tfidf__min_df": [7],
    "tfidf__max_df": [0.7],

    # Logistic Regression params
    "clf__C": [1],
    "clf__penalty": ["l2"],
    "clf__solver": ["liblinear"]   
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring="f1_macro",
    refit = True )

# fit on train only
grid.fit(X_train_full, y_train_full)

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

# evaluate best pipeline on val
best_pipe = grid.best_estimator_

# refit on train + val before final test evaluation
best_pipe.fit(X_train_full, y_train_full)

y_test_pred = best_pipe.predict(X_test_nosw)
print("Test weighted F1:", f1_score(y_test, y_test_pred, average="weighted"))
print("Test accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# feature importance from the fitted pipeline
tfidf = best_pipe.named_steps["tfidf"]
clf = best_pipe.named_steps["clf"]

feature_names = tfidf.get_feature_names_out()
coefs = clf.coef_[0]   # binary case

top_n = 10
top_female_idx = coefs.argsort()[::-1][:top_n]
top_male_idx = coefs.argsort()[:top_n]

print("\nTop female associated features:")
for idx in top_female_idx:
    print(feature_names[idx], coefs[idx])

print("\nTop male associated features:")
for idx in top_male_idx:
    print(feature_names[idx], coefs[idx])
print("==============================")


# In[117]:


# SUPPORT VECTOR MACHINE
X_train = train['response']       
y_train = train['gender']     

X_val = val['response']
y_val = val['gender']

X_test = test['response']       
y_test = test['gender']   

X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

print("==============================")
print("SUPPORT VECTOR MACHINE")


# In[118]:


from sklearn.svm import LinearSVC
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        # keep stopwords for stylistic signals
        stop_words=None
    )),
    ("clf", LinearSVC())
])


param_grid = {
    # TF IDF parameters
    "tfidf__ngram_range": [(1, 2)],
    "tfidf__min_df": [1],
    "tfidf__max_df": [0.95],

    # SVM parameters
    "clf__C": [1],
    "clf__loss": ["hinge"],
    "clf__penalty": ["l2"]
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="f1_macro"   # or "accuracy"
)

grid.fit(X_train_full, y_train_full)

print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

best_svm = grid.best_estimator_

y_pred = best_svm.predict(X_test)

print("Test weighted F1:", f1_score(y_test, y_pred, average="weighted"))
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))


# feature importance from the fitted pipeline
tfidf = best_svm.named_steps["tfidf"]
clf = best_svm.named_steps["clf"]

feature_names = tfidf.get_feature_names_out()
coefs = clf.coef_[0]   # binary case

top_n = 10
top_female_idx = coefs.argsort()[::-1][:top_n]
top_male_idx = coefs.argsort()[:top_n]

print("\nTop female associated features:")
for idx in top_female_idx:
    print(feature_names[idx], coefs[idx])

print("\nTop male associated features:")
for idx in top_male_idx:
    print(feature_names[idx], coefs[idx])


# In[ ]:


# NEURAL NETWORK
pipe_nn = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words=None   # keep stopwords for stylistic signals
    )),
    ("clf", MLPClassifier(
        max_iter=300,
        random_state=42
    ))
])

param_grid_nn = {
    # TF IDF parameters
    "tfidf__ngram_range": [(1,3)],
    "tfidf__min_df": [1],
    "tfidf__max_df": [0.95],

    # Neural network parameters
    "clf__hidden_layer_sizes": [(50, 50)],
    "clf__activation": ["tanh"],
    "clf__alpha": [0.01],      
    "clf__learning_rate_init": [0.001] 
}


grid_nn = GridSearchCV(
    pipe_nn,
    param_grid=param_grid_nn,
    cv=5,
    scoring="f1_macro",
    verbose=2
)

grid_nn.fit(X_train_full, y_train_full)

print("Best parameters (NN):", grid_nn.best_params_)
print("Best CV score (NN):", grid_nn.best_score_)

best_nn = grid_nn.best_estimator_
y_pred_nn = best_nn.predict(X_test)

print("\nTest accuracy (NN):", accuracy_score(y_test, y_pred_nn))
print("\nClassification report (NN):")
print(classification_report(y_test, y_pred_nn))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




