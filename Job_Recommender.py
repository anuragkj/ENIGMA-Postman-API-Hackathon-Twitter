from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
import tweepy
import re
import string
import nltk
from unidecode import unidecode
import csv
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Downloading stopwords and punkt from Natural language toolkit
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Functions for pre-processing of the data that is remove urls, punctuations, numbers etc.
def replace_sep(text):
    text = text.replace("|||", ' ')
    return text


def remove_url(text):
    text = re.sub(r'https?:*?[\s+]', '', text)
    return text


def remove_punct(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text


def remove_numbers(text):
    text = re.sub(r'[0-9]', '', text)
    return text


def convert_lower(text):
    text = text.lower()
    return text


def extra(text):
    text = text.replace("  ", " ")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text


# Using nltk stop words to remove common words not required in processing like a, an the etc.
Stopwords = set(stopwords.words("english"))


def stop_words(text):
    tweet_tokens = word_tokenize(text)
    filtered_words = [w for w in tweet_tokens if not w in Stopwords]
    return " ".join(filtered_words)


# Applying lemmatization i.e. grouping together the words to analyze as one.
def lemmantization(text):
    tokenized_text = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(a) for a in tokenized_text])
    return text


# Doing the pre-processing of data by the functions defined above
def pre_process(text):
    text = replace_sep(text)
    text = remove_url(text)
    text = remove_punct(text)
    text = remove_numbers(text)
    text = convert_lower(text)
    text = extra(text)
    text = stop_words(text)
    text = lemmantization(text)
    return text


# tokenizing the data we retrieve from youtube. Defining the various emojis and emoticons and creating their regex patterns.
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


# Pre processing the tokenized data

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


# Using unidecode to remove all the non ascii characters from our string.
def preproc(s):
    # s=emoji_pattern.sub(r'', s) # no emoji
    s = unidecode(s)
    POSTagger = preprocess(s)
    # print(POSTagger)

    tweet = ' '.join(POSTagger)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    # filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in POSTagger:
        if w not in stop_words:
            filtered_sentence.append(w)
    # print(word_tokens)
    # print(filtered_sentence)
    stemmed_sentence = []
    stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    for w in filtered_sentence:
        stemmed_sentence.append(stemmer2.stem(w))
    # print(stemmed_sentence)

    temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation)
    preProcessed = temp.split(" ")
    final = []
    for i in preProcessed:
        if i not in final:
            if i.isdigit():
                pass
            else:
                if 'http' not in i:
                    final.append(i)
    temp1 = ' '.join(c for c in final)
    # print(preProcessed)
    return temp1


# Using API call to get the tweets of the desired handle
def getTweets(user):
    csvFile = open(resource_path('Resource_Images/user.csv'), 'w', newline='')
    csvWriter = csv.writer(csvFile)
    try:
        for i in range(0, 4):
            tweets = api.user_timeline(screen_name=user, count=1000, include_rts=True, page=i)
            for status in tweets:
                tw = preproc(status.text)
                if tw.find(" ") == -1:
                    tw = "blank"
                csvWriter.writerow([tw])
    except tweepy.TweepyException:
        print("Failed to run the command on that user, Skipping...")
    csvFile.close()


import tweepy as tw

# Declaring secrets and tokens for our twitter api
consumer_key = 'DadKR3DKcG1PWvyh8igvAIaYN'
consumer_secret = 'KjWoOAwm7uwwT0vTGWcuomuPq9Wglo5pA29kPxhOPvddMmO2Eg'
access_token = '1266720191502680066-YGjG1jvAjIOOsG6NibYCDH7trAznfk'
access_token_secret = '1E8fi5w2hi8eRyvnVvHcJnOU9p66oiJaXe5l1PAIQNYqA'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


def join(text):
    return "||| ".join(text)


# For fetching all the tweets from the specified handle
def twits(handle):
    res = api.user_timeline(screen_name=handle, count=1000, include_rts=True)
    tweets = [tweet.text for tweet in res]
    return tweets


# All the info for the processing is loaded. The tweets and frequency saved in their respective files are loaded.
# Vectorizer is defined and the models loaded. The model is fitted to provide the result and on the basis of result the personality is predicted.
# I/E, S/N, T/Fand P/J is chosen to get the personality. These letters are chosen on the basis of higher frequency.
def twit(handle):
    getTweets(handle)
    with open(resource_path('Resource_Images/user.csv'), 'rt') as f:
        csvReader = csv.reader(f)
        tweetList = [rows[0] for rows in csvReader]
    with open(resource_path('Resource_Images/newfrequency300.csv'), 'rt') as f:
        csvReader = csv.reader(f)
        mydict = {rows[1]: int(rows[0]) for rows in csvReader}

    vectorizer = TfidfVectorizer(vocabulary=mydict, min_df=1)
    x = vectorizer.fit_transform(tweetList).toarray()
    df = pd.DataFrame(x)

    model_IE = pickle.load(open(resource_path("Resource_Images/BNIEFinal.sav"), 'rb'))
    model_SN = pickle.load(open(resource_path("Resource_Images/BNSNFinal.sav"), 'rb'))
    model_TF = pickle.load(open(resource_path('Resource_Images/BNTFFinal.sav'), 'rb'))
    model_PJ = pickle.load(open(resource_path('Resource_Images/BNPJFinal.sav'), 'rb'))

    answer = []
    IE = model_IE.predict(df)
    SN = model_SN.predict(df)
    TF = model_TF.predict(df)
    PJ = model_PJ.predict(df)

    b = Counter(IE)
    value = b.most_common(1)

    if value[0][0] == 1.0:
        answer.append("I")
    else:
        answer.append("E")

    b = Counter(SN)
    value = b.most_common(1)

    if value[0][0] == 1.0:
        answer.append("S")
    else:
        answer.append("N")

    b = Counter(TF)
    value = b.most_common(1)

    if value[0][0] == 1:
        answer.append("T")
    else:
        answer.append("F")

    b = Counter(PJ)
    value = b.most_common(1)

    if value[0][0] == 1:
        answer.append("P")
    else:
        answer.append("J")
    mbti = "".join(answer)
    return mbti


def split(text):
    return [char for char in text]


# Listing all the jobs on the basis of features of the personality extracted. These will be clubbed together and returned to get the entire list.
List_jobs_I = ['Accounting manager',
               'Landscape designer',
               'Behavioral therapist',
               'Graphic designer',
               'IT manager']

List_jobs_E = ['Flight attendant',
               'Event planner',
               'Teacher',
               'Criminal investigator',
               'General manager']

List_jobs_S = ['Home health aide',
               'Detective',
               'Actor',
               'Nurse']

List_jobs_N = ['Social worker',
               'HR manager',
               'Counselor',
               'Therapist']

List_jobs_F = ['Entertainer',
               'Mentor',
               'Advocate',
               'Artist',
               'Defender',
               'Dreamer']

List_jobs_T = ['Video game designer',
               'Graphic designer',
               'Social media manager',
               'Copywriter',
               'Public relations manager',
               'Digital marketers',
               'Lawyer',
               'Research scientist',
               'User experience designer',
               'Software architect']

List_jobs_J = ['Showroom designer',
               'IT administrator',
               'Marketing director',
               'Judge',
               'Coach']

List_jobs_P = ['Museum curator',
               'Copywriter',
               'Public relations specialist',
               'Social worker',
               'Medical researcher',
               'Office Manager']

# SImilar to above all the characters are mapped to the respective personality detected
List_ch_I = ['Reflective',
             'Self-aware',
             'Take time making decisions',
             'Feel comfortable being alone',
             'Dont like group works']

List_ch_E = ['Enjoy social settings',
             'Do not like or need a lot of alone time',
             'Thrive around people',
             'Outgoing and optimistic',
             'Prefer to talk out problem or questions']

List_ch_N = ['Listen to and obey their inner voice',
             'Pay attention to their inner dreams',
             'Typically optimistic souls',
             'Strong sense of purpose',
             'Closely observe their surroundings']

List_ch_S = ['Remember events as snapshots of what actually happened',
             'Solve problems by working through facts',
             'Programmatic',
             'Start with facts and then form a big picture',
             'Trust experience first and trust words and symbols less',
             'Sometimes pay so much attention to facts, either present or past, that miss new possibilities']

List_ch_F = ['Decides with heart',
             'Dislikes conflict',
             'Passionate',
             'Driven by emotion',
             'Gentle',
             'Easily hurt',
             'Empathetic',
             'Caring of others']

List_ch_T = ['Logical',
             'Objective',
             'Decides with head',
             'Wants truth',
             'Rational',
             'Impersonal',
             'Critical',
             'Firm with people']

List_ch_J = ['Self-disciplined',
             'Decisive',
             'Structured',
             'Organized',
             'Responsive',
             'Fastidious',
             'Create short and long-term plans',
             'Make a list of things to do',
             'Schedule things in advance',
             'Form and express judgments',
             'Bring closure to an issue so that we can move on']

List_ch_P = ['Relaxed',
             'Adaptable',
             'Non judgemental',
             'Carefree',
             'Creative',
             'Curious',
             'Postpone decisions to see what other options are available',
             'Act spontaneously',
             'Decide what to do as we do it, rather than forming a plan ahead of time',
             'Do things at the last minute']


# Joins and returns the list of characters speific to the the personality detected.
def charcter(text):
    o = split(text)
    characteristics = []
    for i in range(0, 4):
        if o[i] == 'I':
            characteristics.append('\n'.join(List_ch_I))
        if o[i] == 'E':
            characteristics.append('\n'.join(List_ch_E))
        if o[i] == 'N':
            characteristics.append('\n'.join(List_ch_N))
        if o[i] == 'S':
            characteristics.append('\n'.join(List_ch_S))
        if o[i] == 'F':
            characteristics.append('\n'.join(List_ch_F))
        if o[i] == 'T':
            characteristics.append('\n'.join(List_ch_F))
        if o[i] == 'J':
            characteristics.append('\n'.join(List_ch_J))
        if o[i] == 'P':
            characteristics.append('\n'.join(List_ch_P))
    crct = '\n'.join(characteristics)
    data = crct.split("\n")
    return data


# Joins and returns the list of job recommendations speific to the the personality detected.
def recomend(text):
    b = split(text)
    jobs = []
    for i in range(0, 4):
        if b[i] == 'I':
            jobs.append('\n'.join(List_jobs_I))
        if b[i] == 'E':
            jobs.append('\n'.join(List_jobs_E))
        if b[i] == 'N':
            jobs.append('\n'.join(List_jobs_N))
        if b[i] == 'S':
            jobs.append('\n'.join(List_jobs_S))
        if b[i] == 'F':
            jobs.append('\n'.join(List_jobs_F))
        if b[i] == 'T':
            jobs.append('\n'.join(List_jobs_T))
        if b[i] == 'J':
            jobs.append('\n'.join(List_jobs_J))
        if b[i] == 'P':
            jobs.append('\n'.join(List_jobs_P))
    crct1 = '\n'.join(jobs)
    data1 = crct1.split("\n")
    return (split(data1))


def pp(handle):
    personality = twit(handle)
    return personality, recomend(personality), charcter(personality)


def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Creating the Tkinter frontend for proper interaction
from tkinter import *
from PIL import ImageTk
import tkinter as tk


class MyWindow:
    def __init__(self, win):
        self.bg1 = None
        self.D_lbl0 = ttk.Label(win, text='Personality Based Job Recommender Using Twitter Data', font=("Arial", 10))
        self.D_lbl0.place(x=40, y=30)
        self.btn1 = ttk.Button(win, text='Start Application', style='Accent.TButton', command=self.home1)
        self.btn1.place(x=130, y=120)
        self.btn1 = ttk.Button(win, text='Quit', style='Accent.TButton', command=win.destroy)
        self.btn1.place(x=142, y=165)

    def mbti(self):
        newwin = Toplevel(window)
        newwin.geometry("1280x720")
        self.btn1 = ttk.Button(newwin, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [newwin.destroy(), self.mbti()])
        self.btn1.place(x=510, y=120)
        self.btn1 = ttk.Button(newwin, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin.destroy()])
        self.btn1.place(x=410, y=120)
        self.btn1 = ttk.Button(newwin, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [self.mbt(), newwin.destroy()])
        self.btn1.place(x=610, y=120)
        self.btn1 = ttk.Button(newwin, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [self.explore(), newwin.destroy()])
        self.btn1.place(x=710, y=120)
        self.bg1 = ImageTk.PhotoImage(file=resource_path("Resource_Images/data_info.png"))
        canvas = Canvas(newwin, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        canvas.create_image(1050, 70, image=self.bg1, anchor="ne")

        self.D_lbl0 = ttk.Label(newwin, text='Personality Based Job Recommender Using Twitter Data', font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)

    def mbt(self):
        newwin1 = Toplevel(window)
        newwin1.geometry("1280x720")
        self.D_lbl0 = ttk.Label(newwin1, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.btn1 = ttk.Button(newwin1, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), newwin1.destroy()])
        self.btn1.place(x=510, y=120)
        self.btn1 = ttk.Button(newwin1, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin1.destroy()])
        self.btn1.place(x=410, y=120)
        self.btn1 = ttk.Button(newwin1, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [newwin1.destroy(), self.mbt()])
        self.btn1.place(x=610, y=120)
        self.btn1 = ttk.Button(newwin1, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [self.explore(), newwin1.destroy()])
        self.btn1.place(x=710, y=120)
        canvas = Canvas(newwin1, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        self.bg1 = ImageTk.PhotoImage(file=resource_path("Resource_Images/TestResults.png"))
        canvas.create_image(1050, 70, image=self.bg1, anchor="ne")

    def explore(self):
        newwin2 = Toplevel(window)
        newwin2.geometry("1280x720")
        canvas = Canvas(newwin2, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file=resource_path("Resource_Images/CountPlot.png"))
        canvas.create_image(50, 70, image=self.bg1, anchor="nw")

        self.D_lbl0 = ttk.Label(newwin2, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.btn1 = ttk.Button(newwin2, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), newwin2.destroy()])
        self.btn1.place(x=510, y=120)
        self.btn1 = ttk.Button(newwin2, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin2.destroy()])
        self.btn1.place(x=410, y=120)
        self.btn1 = ttk.Button(newwin2, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [self.mbt(), newwin2.destroy()])
        self.btn1.place(x=610, y=120)
        self.btn1 = ttk.Button(newwin2, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [newwin2.destroy(), self.explore()])
        self.btn1.place(x=710, y=120)
        self.btn1 = ttk.Button(newwin2, text='PIE PLOT', style='Accent.TButton',
                               command=lambda: [self.explore1(), newwin2.destroy()])
        self.btn1.place(x=1000, y=200)
        self.btn1 = ttk.Button(newwin2, text='DIS PLOT', style='Accent.TButton',
                               command=lambda: [self.explore2(), newwin2.destroy()])
        self.btn1.place(x=1000, y=250)
        self.btn1 = ttk.Button(newwin2, text='I-E PLOT', style='Accent.TButton',
                               command=lambda: [self.explore3(), newwin2.destroy()])
        self.btn1.place(x=1000, y=300)
        self.btn1 = ttk.Button(newwin2, text='N-S PLOT', style='Accent.TButton',
                               command=lambda: [self.explore4(), newwin2.destroy()])
        self.btn1.place(x=1000, y=350)
        self.btn1 = ttk.Button(newwin2, text='T-F PLOT', style='Accent.TButton',
                               command=lambda: [self.explore5(), newwin2.destroy()])
        self.btn1.place(x=1000, y=400)
        self.btn1 = ttk.Button(newwin2, text='P-J PLOT', style='Accent.TButton',
                               command=lambda: [self.explore6(), newwin2.destroy()])
        self.btn1.place(x=1000, y=450)

    def explore1(self):
        newwin3 = Toplevel(window)
        newwin3.geometry("1280x720")
        canvas = Canvas(newwin3, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file=resource_path("Resource_Images/PiePlot.png"))
        canvas.create_image(50, 70, image=self.bg1, anchor="nw")

        self.D_lbl0 = ttk.Label(newwin3, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.btn1 = ttk.Button(newwin3, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), newwin3.destroy()])
        self.btn1.place(x=510, y=120)
        self.btn1 = ttk.Button(newwin3, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin3.destroy()])
        self.btn1.place(x=410, y=120)
        self.btn1 = ttk.Button(newwin3, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [self.mbt(), newwin3.destroy()])
        self.btn1.place(x=610, y=120)
        self.btn1 = ttk.Button(newwin3, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [newwin3.destroy(), self.explore()])
        self.btn1.place(x=710, y=120)
        self.btn1 = ttk.Button(newwin3, text='PIE PLOT', style='Accent.TButton',
                               command=lambda: [self.explore1(), newwin3.destroy()])
        self.btn1.place(x=1000, y=200)
        self.btn1 = ttk.Button(newwin3, text='DIS PLOT', style='Accent.TButton',
                               command=lambda: [self.explore2(), newwin3.destroy()])
        self.btn1.place(x=1000, y=250)
        self.btn1 = ttk.Button(newwin3, text='I-E PLOT', style='Accent.TButton',
                               command=lambda: [self.explore3(), newwin3.destroy()])
        self.btn1.place(x=1000, y=300)
        self.btn1 = ttk.Button(newwin3, text='N-S PLOT', style='Accent.TButton',
                               command=lambda: [self.explore4(), newwin3.destroy()])
        self.btn1.place(x=1000, y=350)
        self.btn1 = ttk.Button(newwin3, text='T-F PLOT', style='Accent.TButton',
                               command=lambda: [self.explore5(), newwin3.destroy()])
        self.btn1.place(x=1000, y=400)
        self.btn1 = ttk.Button(newwin3, text='P-J PLOT', style='Accent.TButton',
                               command=lambda: [self.explore6(), newwin3.destroy()])
        self.btn1.place(x=1000, y=450)

    def explore2(self):
        newwin4 = Toplevel(window)
        newwin4.geometry("1280x720")
        canvas = Canvas(newwin4, width=2500, height=2000)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file=resource_path("Resource_Images/Displot.png"))
        canvas.create_image(50, 70, image=self.bg1, anchor="nw")

        self.D_lbl0 = ttk.Label(newwin4, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.btn1 = ttk.Button(newwin4, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), newwin4.destroy()])
        self.btn1.place(x=510, y=120)
        self.btn1 = ttk.Button(newwin4, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin4.destroy()])
        self.btn1.place(x=410, y=120)
        self.btn1 = ttk.Button(newwin4, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [self.mbt(), newwin4.destroy()])
        self.btn1.place(x=610, y=120)
        self.btn1 = ttk.Button(newwin4, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [newwin4.destroy(), self.explore()])
        self.btn1.place(x=710, y=120)
        self.btn1 = ttk.Button(newwin4, text='PIE PLOT', style='Accent.TButton',
                               command=lambda: [self.explore1(), newwin4.destroy()])
        self.btn1.place(x=1000, y=200)
        self.btn1 = ttk.Button(newwin4, text='DIS PLOT', style='Accent.TButton',
                               command=lambda: [self.explore2(), newwin4.destroy()])
        self.btn1.place(x=1000, y=250)
        self.btn1 = ttk.Button(newwin4, text='I-E PLOT', style='Accent.TButton',
                               command=lambda: [self.explore3(), newwin4.destroy()])
        self.btn1.place(x=1000, y=300)
        self.btn1 = ttk.Button(newwin4, text='N-S PLOT', style='Accent.TButton',
                               command=lambda: [self.explore4(), newwin4.destroy()])
        self.btn1.place(x=1000, y=350)
        self.btn1 = ttk.Button(newwin4, text='T-F PLOT', style='Accent.TButton',
                               command=lambda: [self.explore5(), newwin4.destroy()])
        self.btn1.place(x=1000, y=400)
        self.btn1 = ttk.Button(newwin4, text='P-J PLOT', style='Accent.TButton',
                               command=lambda: [self.explore6(), newwin4.destroy()])
        self.btn1.place(x=1000, y=450)

    def explore3(self):
        newwin5 = Toplevel(window)
        newwin5.geometry("1280x720")
        canvas = Canvas(newwin5, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file=resource_path("Resource_Images/I_E.png"))
        canvas.create_image(50, 70, image=self.bg1, anchor="nw")

        self.D_lbl0 = ttk.Label(newwin5, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.btn1 = ttk.Button(newwin5, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), newwin5.destroy()])
        self.btn1.place(x=510, y=120)
        self.btn1 = ttk.Button(newwin5, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin5.destroy()])
        self.btn1.place(x=410, y=120)
        self.btn1 = ttk.Button(newwin5, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [self.mbt(), newwin5.destroy()])
        self.btn1.place(x=610, y=120)
        self.btn1 = ttk.Button(newwin5, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [newwin5.destroy(), self.explore()])
        self.btn1.place(x=710, y=120)
        self.btn1 = ttk.Button(newwin5, text='PIE PLOT', style='Accent.TButton',
                               command=lambda: [self.explore1(), newwin5.destroy()])
        self.btn1.place(x=1000, y=200)
        self.btn1 = ttk.Button(newwin5, text='DIS PLOT', style='Accent.TButton',
                               command=lambda: [self.explore2(), newwin5.destroy()])
        self.btn1.place(x=1000, y=250)
        self.btn1 = ttk.Button(newwin5, text='I-E PLOT', style='Accent.TButton',
                               command=lambda: [self.explore3(), newwin5.destroy()])
        self.btn1.place(x=1000, y=300)
        self.btn1 = ttk.Button(newwin5, text='N-S PLOT', style='Accent.TButton',
                               command=lambda: [self.explore4(), newwin5.destroy()])
        self.btn1.place(x=1000, y=350)
        self.btn1 = ttk.Button(newwin5, text='T-F PLOT', style='Accent.TButton',
                               command=lambda: [self.explore5(), newwin5.destroy()])
        self.btn1.place(x=1000, y=400)
        self.btn1 = ttk.Button(newwin5, text='P-J PLOT', style='Accent.TButton',
                               command=lambda: [self.explore6(), newwin5.destroy()])
        self.btn1.place(x=1000, y=450)

    def explore4(self):
        newwin6 = Toplevel(window)
        newwin6.geometry("1280x720")
        canvas = Canvas(newwin6, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file=resource_path("Resource_Images/N_S.png"))
        canvas.create_image(50, 70, image=self.bg1, anchor="nw")

        self.D_lbl0 = ttk.Label(newwin6, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.btn1 = ttk.Button(newwin6, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), newwin6.destroy()])
        self.btn1.place(x=510, y=120)
        self.btn1 = ttk.Button(newwin6, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin6.destroy()])
        self.btn1.place(x=410, y=120)
        self.btn1 = ttk.Button(newwin6, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [self.mbt(), newwin6.destroy()])
        self.btn1.place(x=610, y=120)
        self.btn1 = ttk.Button(newwin6, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [newwin6.destroy(), self.explore()])
        self.btn1.place(x=710, y=120)
        self.btn1 = ttk.Button(newwin6, text='PIE PLOT', style='Accent.TButton',
                               command=lambda: [self.explore1(), newwin6.destroy()])
        self.btn1.place(x=1000, y=200)
        self.btn1 = ttk.Button(newwin6, text='DIS PLOT', style='Accent.TButton',
                               command=lambda: [self.explore2(), newwin6.destroy()])
        self.btn1.place(x=1000, y=250)
        self.btn1 = ttk.Button(newwin6, text='I-E PLOT', style='Accent.TButton',
                               command=lambda: [self.explore3(), newwin6.destroy()])
        self.btn1.place(x=1000, y=300)
        self.btn1 = ttk.Button(newwin6, text='N-S PLOT', style='Accent.TButton',
                               command=lambda: [self.explore4(), newwin6.destroy()])
        self.btn1.place(x=1000, y=350)
        self.btn1 = ttk.Button(newwin6, text='T-F PLOT', style='Accent.TButton',
                               command=lambda: [self.explore5(), newwin6.destroy()])
        self.btn1.place(x=1000, y=400)
        self.btn1 = ttk.Button(newwin6, text='P-J PLOT', style='Accent.TButton',
                               command=lambda: [self.explore6(), newwin6.destroy()])
        self.btn1.place(x=1000, y=450)

    def explore5(self):
        newwin7 = Toplevel(window)
        newwin7.geometry("1280x720")
        canvas = Canvas(newwin7, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file=resource_path("Resource_Images/T_F.png"))
        canvas.create_image(50, 70, image=self.bg1, anchor="nw")

        self.D_lbl0 = ttk.Label(newwin7, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.btn1 = ttk.Button(newwin7, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), newwin7.destroy()])
        self.btn1.place(x=510, y=120)
        self.btn1 = ttk.Button(newwin7, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin7.destroy()])
        self.btn1.place(x=410, y=120)
        self.btn1 = ttk.Button(newwin7, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [self.mbt(), newwin7.destroy()])
        self.btn1.place(x=610, y=120)
        self.btn1 = ttk.Button(newwin7, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [newwin7.destroy(), self.explore()])
        self.btn1.place(x=710, y=120)
        self.btn1 = ttk.Button(newwin7, text='PIE PLOT', style='Accent.TButton',
                               command=lambda: [self.explore1(), newwin7.destroy()])
        self.btn1.place(x=1000, y=200)
        self.btn1 = ttk.Button(newwin7, text='DIS PLOT', style='Accent.TButton',
                               command=lambda: [self.explore2(), newwin7.destroy()])
        self.btn1.place(x=1000, y=250)
        self.btn1 = ttk.Button(newwin7, text='I-E PLOT', style='Accent.TButton',
                               command=lambda: [self.explore3(), newwin7.destroy()])
        self.btn1.place(x=1000, y=300)
        self.btn1 = ttk.Button(newwin7, text='N-S PLOT', style='Accent.TButton',
                               command=lambda: [self.explore4(), newwin7.destroy()])
        self.btn1.place(x=1000, y=350)
        self.btn1 = ttk.Button(newwin7, text='T-F PLOT', style='Accent.TButton',
                               command=lambda: [self.explore5(), newwin7.destroy()])
        self.btn1.place(x=1000, y=400)
        self.btn1 = ttk.Button(newwin7, text='P-J PLOT', style='Accent.TButton',
                               command=lambda: [self.explore6(), newwin7.destroy()])
        self.btn1.place(x=1000, y=450)

    def explore6(self):
        newwin8 = Toplevel(window)
        newwin8.geometry("1280x720")
        canvas = Canvas(newwin8, width=2500, height=2000)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=100)
        self.bg1 = ImageTk.PhotoImage(file=resource_path("Resource_Images/J_P.png"))
        canvas.create_image(50, 70, image=self.bg1, anchor="nw")

        self.D_lbl0 = ttk.Label(newwin8, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.btn1 = ttk.Button(newwin8, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), newwin8.destroy()])
        self.btn1.place(x=510, y=120)
        self.btn1 = ttk.Button(newwin8, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin8.destroy()])
        self.btn1.place(x=410, y=120)
        self.btn1 = ttk.Button(newwin8, text='MBTI TEST', style='Accent.TButton',
                               command=lambda: [self.mbt(), newwin8.destroy()])
        self.btn1.place(x=610, y=120)
        self.btn1 = ttk.Button(newwin8, text='EXPLORATORY DATA', style='Accent.TButton',
                               command=lambda: [newwin8.destroy(), self.explore()])
        self.btn1.place(x=710, y=120)
        self.btn1 = ttk.Button(newwin8, text='PIE PLOT', style='Accent.TButton',
                               command=lambda: [self.explore1(), newwin8.destroy()])
        self.btn1.place(x=1000, y=200)
        self.btn1 = ttk.Button(newwin8, text='DIS PLOT', style='Accent.TButton',
                               command=lambda: [self.explore2(), newwin8.destroy()])
        self.btn1.place(x=1000, y=250)
        self.btn1 = ttk.Button(newwin8, text='I-E PLOT', style='Accent.TButton',
                               command=lambda: [self.explore3(), newwin8.destroy()])
        self.btn1.place(x=1000, y=300)
        self.btn1 = ttk.Button(newwin8, text='N-S PLOT', style='Accent.TButton',
                               command=lambda: [self.explore4(), newwin8.destroy()])
        self.btn1.place(x=1000, y=350)
        self.btn1 = ttk.Button(newwin8, text='T-F PLOT', style='Accent.TButton',
                               command=lambda: [self.explore5(), newwin8.destroy()])
        self.btn1.place(x=1000, y=400)
        self.btn1 = ttk.Button(newwin8, text='P-J PLOT', style='Accent.TButton',
                               command=lambda: [self.explore6(), newwin8.destroy()])
        self.btn1.place(x=1000, y=450)

    def twitter(self):
        newwin9 = Toplevel(window)
        newwin9.geometry("1280x720")
        self.D_lbl0 = ttk.Label(newwin9, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.btn1 = ttk.Button(newwin9, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin9.destroy()])
        self.btn1.place(x=365, y=120)
        self.btn1 = ttk.Button(newwin9, text='TWITTER POSTS', style='Accent.TButton',
                               command=lambda: [self.posts(), newwin9.destroy()])
        self.btn1.place(x=465, y=120)
        self.btn1 = ttk.Button(newwin9, text='PREDICT PERSONALITY', style='Accent.TButton',
                               command=lambda: [self.home(), newwin9.destroy()])
        self.btn1.place(x=597, y=120)
        self.btn1 = ttk.Button(newwin9, text='RECOMMENDATIONS', style='Accent.TButton',
                               command=lambda: [self.recomends(), newwin9.destroy()])
        self.btn1.place(x=775, y=120)

    def posts(self):
        newwin10 = Toplevel(window)
        newwin10.geometry("1280x720")
        self.D_btn1 = ttk.Button(newwin10, text='TWITTER POSTS', style='Accent.TButton',
                                 command=lambda: [newwin10.destroy(), self.posts()])
        self.D_btn1.place(x=465, y=120)
        self.D_b1 = ttk.Button(newwin10, text='PREDICT PERSONALITY', style='Accent.TButton',
                               command=lambda: [self.home(), newwin10.destroy()])
        self.D_b1.place(x=597, y=120)
        self.D_btn1 = ttk.Button(newwin10, text='RECOMMENDATIONS', style='Accent.TButton',
                                 command=lambda: [self.recomends(), newwin10.destroy()])
        self.D_btn1.place(x=775, y=120)
        self.btn1 = ttk.Button(newwin10, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin10.destroy()])
        self.btn1.place(x=365, y=120)
        self.D_lbl0 = ttk.Label(newwin10, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.t1 = Text(newwin10)
        self.t3 = Text(newwin10)
        self.t2 = ttk.Entry(newwin10)
        self.lbl1 = ttk.Label(newwin10, text='Enter Twitter ID: ')
        self.lbl1.place(x=500, y=180)
        self.lbl4 = ttk.Label(newwin10, text='Tweets data of user:')
        self.lbl4.place(x=50, y=290)
        self.lbl4 = ttk.Label(newwin10, text='Cleaned data:')
        self.lbl4.place(x=680, y=290)
        self.t1.place(x=50, y=320)
        self.t3.place(x=680, y=320)
        self.t2.place(x=600, y=170, height=45)
        self.b1 = ttk.Button(newwin10, text='Get Tweets', style='Accent.TButton', command=self.twt)
        self.b1.place(x=250, y=240, width=130, height=50)
        self.b1 = ttk.Button(newwin10, text='Pre Process Tweets', style='Accent.TButton', command=self.twt1)
        self.b1.place(x=850, y=240, width=170, height=50)

    def twt(self):
        handle = self.t2.get()
        res = twits(handle)
        self.t1.configure(state='normal')
        self.t1.delete('1.0', END)

        self.t1.insert(END, str(res))
        self.t1.configure(state='disabled')

    def twt1(self):
        handle = self.t2.get()
        res1 = twits(handle)
        tx1 = join(res1)
        tx2 = pre_process(tx1)
        self.t3.configure(state='normal')
        self.t3.delete('1.0', END)

        self.t3.insert(END, str(tx2))
        self.t3.configure(state='disabled')

    def recomends(self):
        newwin11 = Toplevel(window)
        newwin11.geometry("1280x720")
        self.D_btn1 = ttk.Button(newwin11, text='TWITTER POSTS', style='Accent.TButton',
                                 command=lambda: [self.posts(), newwin11.destroy()])
        self.D_btn1.place(x=465, y=120)
        self.D_b1 = ttk.Button(newwin11, text='PREDICT PERSONALITY', style='Accent.TButton',
                               command=lambda: [self.home(), newwin11.destroy()])
        self.D_b1.place(x=597, y=120)
        self.D_btn1 = ttk.Button(newwin11, text='RECOMMENDATIONS', style='Accent.TButton',
                                 command=lambda: [newwin11.destroy(), self.recomends()])
        self.D_btn1.place(x=775, y=120)
        self.btn1 = ttk.Button(newwin11, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin11.destroy()])
        self.btn1.place(x=365, y=120)
        self.D_lbl0 = ttk.Label(newwin11, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.lbl1 = ttk.Label(newwin11, text='Enter handle name:')
        self.lbl2 = ttk.Label(newwin11, text='Job Recommendations:')
        self.lbl5 = ttk.Label(newwin11, text='Personality Type:')
        self.b1 = ttk.Button(newwin11, text='Recommendations', style='Accent.TButton', command=self.recmd)
        self.b1.place(x=600, y=280)
        self.t0 = ttk.Entry(newwin11)
        self.t2 = Text(newwin11, height=15, width=85)
        self.t1 = Text(newwin11, height=2, width=10)
        self.t0.place(x=650, y=220, height=40)
        self.lbl2.place(x=400, y=410)
        self.lbl1.place(x=485, y=227)
        self.lbl5.place(x=500, y=332)
        self.t1.place(x=650, y=330)
        self.t2.place(x=400, y=460)

        self.i = ttk.Label(newwin11, text='I - Introvert')
        self.i.place(x=100, y=550)
        self.e = ttk.Label(newwin11, text='E - Extrovert')
        self.e.place(x=200, y=550)
        self.n = ttk.Label(newwin11, text='N - Intuitive')
        self.n.place(x=100, y=570)
        self.s = ttk.Label(newwin11, text='S - Sensing')
        self.s.place(x=200, y=570)
        self.f = ttk.Label(newwin11, text='F - Feeling')
        self.f.place(x=100, y=590)
        self.tt = ttk.Label(newwin11, text='T - Thinking')
        self.tt.place(x=200, y=590)
        self.j = ttk.Label(newwin11, text='J - Judging')
        self.j.place(x=100, y=610)
        self.p = ttk.Label(newwin11, text='P - Perceiving')
        self.p.place(x=200, y=610)

    def recmd(self):
        handle = self.t0.get()
        res = twit(handle)
        self.t1.configure(state='normal')
        self.t1.delete('1.0', END)

        self.t1.insert(END, str(res))
        self.t1.configure(state='disabled')
        r = self.t1.get
        result = recomend(res)
        self.t2.configure(state='normal')
        self.t2.delete('1.0', END)

        for i in range(len(result)):
            self.t2.insert(END, str(result[i]))
            self.t2.insert(END, str('\n'))
        self.t2.configure(state='disabled')

    def home(self):
        newwin12 = Toplevel(window)
        newwin12.geometry("1280x720")
        self.D_btn1 = ttk.Button(newwin12, text='TWITTER POSTS', style='Accent.TButton',
                                 command=lambda: [self.posts(), newwin12.destroy()])
        self.D_btn1.place(x=465, y=120)
        self.D_b1 = ttk.Button(newwin12, text='PREDICT PERSONALITY', style='Accent.TButton',
                               command=lambda: [newwin12.destroy(), self.home()])
        self.D_b1.place(x=597, y=120)
        self.D_btn1 = ttk.Button(newwin12, text='RECOMMENDATIONS', style='Accent.TButton',
                                 command=lambda: [self.recomends(), newwin12.destroy()])
        self.D_btn1.place(x=775, y=120)
        self.btn1 = ttk.Button(newwin12, text='HOME', style='Accent.TButton',
                               command=lambda: [self.home1(), newwin12.destroy()])
        self.btn1.place(x=365, y=120)
        self.D_lbl0 = ttk.Label(newwin12, text='Personality Based Job Recommender Using Twitter Data',
                                font=("Arial", 15))
        self.D_lbl0.place(x=400, y=30)
        self.lbl2 = ttk.Label(newwin12, text='Characteristics of Personalities:')
        self.lbl2.place(x=650, y=250)
        self.t = Text(newwin12, height=15, width=85)
        self.t.place(x=650, y=310)
        self.lbl2 = ttk.Label(newwin12, text='Predicted Personality Type')
        self.lbl2.place(x=200, y=437)
        self.lbl1 = ttk.Label(newwin12, text='Enter handle name of twitter:')
        self.lbl1.place(x=150, y=307)
        self.t1 = ttk.Entry(newwin12)
        self.t1.place(x=350, y=300, height=40)
        self.t2 = Text(newwin12, width=10)
        self.t2.place(x=380, y=430, height=40)
        self.b1 = ttk.Button(newwin12, text='Predict Personality', style='Accent.TButton', command=self.predict)
        self.b1.place(x=300, y=375)

        self.i = ttk.Label(newwin12, text='I - Introvert')
        self.i.place(x=100, y=550)
        self.e = ttk.Label(newwin12, text='E - Extrovert')
        self.e.place(x=200, y=550)
        self.n = ttk.Label(newwin12, text='N - Intuitive')
        self.n.place(x=100, y=570)
        self.s = ttk.Label(newwin12, text='S - Sensing')
        self.s.place(x=200, y=570)
        self.f = ttk.Label(newwin12, text='F - Feeling')
        self.f.place(x=100, y=590)
        self.tt = ttk.Label(newwin12, text='T - Thinking')
        self.tt.place(x=200, y=590)
        self.j = ttk.Label(newwin12, text='J - Judging')
        self.j.place(x=100, y=610)
        self.p = ttk.Label(newwin12, text='P - Perceiving')
        self.p.place(x=200, y=610)

    def predict(self):
        handle = self.t1.get()
        res = twit(handle)
        self.t2.configure(state='normal')
        self.t2.delete('1.0', END)

        self.t2.insert(END, str(res))
        self.t2.configure(state='disabled')
        r = self.t2.get
        result = charcter(res)
        self.t.configure(state='normal')
        self.t.delete('1.0', END)

        for i in range(len(result)):
            self.t.insert(END, str(result[i]))
            self.t.insert(END, str('\n'))
        self.t.configure(state='disabled')

    def home1(self):
        newwin13 = Toplevel(window)
        newwin13.geometry("600x600")
        self.bg1 = ImageTk.PhotoImage(file=resource_path("Resource_Images/Home.png"))
        canvas = Canvas(newwin13, width=50, height=60)
        canvas.pack(expand=True, fill=BOTH)
        canvas.pack(padx=0, pady=170)
        canvas.create_image(500, 70, image=self.bg1, anchor="ne")
        self.D_lbl0 = ttk.Label(newwin13, text='Personality Based Job Recommender Using Twitter Data ',
                                font=("Arial", 15))
        self.D_lbl0.place(x=50, y=30)
        self.btn1 = ttk.Button(newwin13, text='MBTI DATA', style='Accent.TButton',
                               command=lambda: [self.mbti(), newwin13.destroy()])
        self.btn1.place(x=150, y=120)
        self.btn1 = ttk.Button(newwin13, text='TWITTER DATA', style='Accent.TButton',
                               command=lambda: [self.twitter(), newwin13.destroy()])
        self.btn1.place(x=350, y=120)


from tkinter import ttk

window = tk.Tk()
window.tk.call('source', resource_path(resource_path('Resource_Images/forest-dark.tcl')))

# Set the theme with the theme_use method
ttk.Style().theme_use('forest-dark')

window.title("ENIGMA")
mywin = MyWindow(window)
window.geometry("400x300")
window.iconphoto(True, tk.PhotoImage(file=resource_path('Resource_Images/icon.png')))
window.mainloop()
