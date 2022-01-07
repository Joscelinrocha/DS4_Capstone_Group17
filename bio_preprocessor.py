import pandas as pd
import re
import gensim
from nltk.stem import WordNetLemmatizer

punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'         # define a string of punctuation symbols
#getting rid of emojis
regex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)

# Functions to clean bios
def remove_links(user_bio):
    """Takes a string and removes web links from it"""
    user_bio = re.sub(r'http\S+', '', user_bio)   # remove http links
    user_bio = re.sub(r'bit.ly/\S+', '', user_bio)  # remove bitly links
    user_bio = user_bio.strip('[link]')   # remove [links]
    user_bio = re.sub(r'pic.twitter\S+','', user_bio)
    return user_bio

def remove_users(user_bio):
    """Takes a string and removes retweet and @user information"""
    user_bio = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', user_bio)  # remove re-tweet
    user_bio = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', user_bio)  # remove tweeted at
    return user_bio

def remove_hashtags(user_bio):
    """Takes a string and removes any hash tags"""
    user_bio = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', user_bio)  # remove hash tags
    return user_bio

def remove_av(user_bio):
    """Takes a string and removes AUDIO/VIDEO tags or labels"""
    user_bio = re.sub('VIDEO:', '', user_bio)  # remove 'VIDEO:' from start
    user_bio = re.sub('AUDIO:', '', user_bio)  # remove 'AUDIO:' from start
    return user_bio

def tokenize(user_bio):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    result = []
    for token in gensim.utils.simple_preprocess(user_bio):
        if token not in gensim.parsing.preprocessing.STOPWORDS \
                and len(token) > 2:  # drops words with less than 3 characters
            result.append(lemmatize(token))
    return result

def lemmatize(token):
    """Returns lemmatization of a token"""
    return WordNetLemmatizer().lemmatize(token, pos='v')

def preprocess_bio(user_bio):
    """Main master function to clean user_bio, stripping noisy characters, and tokenizing use lemmatization"""
    user_bio = remove_users(user_bio)
    user_bio = remove_links(user_bio)
    user_bio = remove_hashtags(user_bio)
    user_bio = remove_av(user_bio)
    user_bio = re.sub(regex_pattern, '', user_bio)
    user_bio = user_bio.lower()  # lower case
    user_bio = re.sub('[' + punctuation + ']+', ' ', user_bio)  # strip punctuation
    user_bio = re.sub('\s+', ' ', user_bio)  # remove double spacing
    user_bio = re.sub('([0-9]+)', '', user_bio)  # remove numbers
    user_bio_token_list = tokenize(user_bio)  # apply lemmatization and tokenization
    user_bio = ' '.join(user_bio_token_list)
    return user_bio

def basic_clean(user_bio):
    """Main master function to clean user_bio only without tokenization or removal of stopwords"""
    user_bio = remove_users(user_bio)
    user_bio = remove_links(user_bio)
    user_bio = remove_hashtags(user_bio)
    user_bio = remove_av(user_bio)
    user_bio = user_bio.lower()  # lower case
    user_bio = re.sub('[' + punctuation + ']+', ' ', user_bio)  # strip punctuation
    user_bio = re.sub('\s+', ' ', user_bio)  # remove double spacing
    user_bio = re.sub('([0-9]+)', '', user_bio)  # remove numbers
    user_bio = re.sub('📝 …', '', user_bio)
    return user_bio

def tokenize_bios(df):
    """Main function to read in and return cleaned and preprocessed dataframe.
    This can be used in Jupyter notebooks by importing this module and calling the tokenize_bios() function

    Args:
        df = data frame object to apply cleaning to

    Returns:
        pandas data frame with cleaned tokens
    """

    df['tokens'] = df.user_bio.apply(preprocess_bio)
    num_bios = len(df)
    print('Complete. Number of bios that have been cleaned and tokenized : {}'.format(num_bios))
    return df