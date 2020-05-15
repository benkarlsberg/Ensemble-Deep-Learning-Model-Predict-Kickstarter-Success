import numpy as np
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords

def stopwords_list():
    sw = set(stopwords.words('english'))
    new_stopwords = []
    for word in sw:
        new_stopwords.append(preprocesser(word))
    return new_stopwords

def preprocesser(doc):
    # lower case
    doc = doc.lower()
    # punctuations
    doc = re.sub('[~*.^!?\'_]*', '', doc)
    # numbers
    doc = re.sub(r'\w*\d\w*', '', doc).strip()
    # lemmatize
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    pos_tagged_text = pos_tag(doc.split())
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

def lemmatize_words(text):
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    pos_tagged_text = pos_tag(text.split())
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

def calculate_threshold_values(prob, y):
    '''
    Build dataframe of the various confusion-matrix ratios by threshold
    from a list of predicted probabilities and actual y values
    '''
    df = pd.DataFrame({'prob': prob, 'y': y})
    df.sort_values('prob', inplace=True) # kind of the biggest deal here in terms of making this ROC curve possible
    
    actual_p = df.y.sum()
    actual_n = df.shape[0] - df.y.sum()

    df['tn'] = (df.y == 0).cumsum()
    df['fn'] = df.y.cumsum()
    df['fp'] = actual_n - df.tn
    df['tp'] = actual_p - df.fn

    df['fpr'] = df.fp/(df.fp + df.tn)
    df['tpr'] = df.tp/(df.tp + df.fn)
    df['precision'] = df.tp/(df.tp + df.fp)
    df = df.reset_index(drop=True)
    return df

def plot_roc(ax, df):
    ax.plot([1]+list(df.fpr), [1]+list(df.tpr), label="ROC", color = 'b')
    ax.plot([0,1],[0,1], 'k', label="random")
    ax.set_xlabel('fpr', fontsize = 15)
    ax.set_ylabel('tpr', fontsize = 15)
    ax.set_title('ROC Curve', fontsize = 20)
    ax.legend()

def plot_precision_recall(ax, df):
    ax.plot(df.tpr,df.precision, label='precision/recall')
    #ax.plot([0,1],[0,1], 'k')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision/Recall Curve')
    ax.plot([0,1],[df.precision[0],df.precision[0]], 'k', label='random')
    ax.set_xlim(xmin=0,xmax=1)
    ax.set_ylim(ymin=0,ymax=1)
