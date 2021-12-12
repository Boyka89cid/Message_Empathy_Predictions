# Importing Sckit-learn, spacy, torch, nltk Libraries
import pandas as pd
import numpy as np
import spacy
import re
import string
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import Dataset
import nltk

# NLTK To Remove Stopwords or for Lemmatization
nltk.download('stopwords')
from nltk.corpus import stopwords

spacy.require_gpu()
tok = spacy.load('en_core_web_sm')
english_stopwords = tok.Defaults.stop_words
stop = stopwords.words('english')
word_lemmatizer = nltk.stem.WordNetLemmatizer()
word_tokenizer = nltk.tokenize.WhitespaceTokenizer()


class DataProcessing(object):
    """
    Class Defined for data cleaning, lemmatization, tokenization, words to vectors from  Glove,
    Multilabel Binarization for Multilabel Encoding, other Data Preprocessing steps and
    Machine Learning Base-line modelling
    """

    def __init__(self, labelled_message_file, empathies_file):
        super(DataProcessing, self).__init__()
        self.empathy_labeled_data = pd.read_csv(labelled_message_file)
        self.empathies = pd.read_csv(empathies_file)
        self.counts = Counter()  # occurences of each word
        self.vocab2index = {"": 0, "UNK": 1}  # Dictionary for corpus word which maps to unique identifier
        self.words = ["", "UNK"]  # List of Corpus words
        self.output_size = None  # Output Size is Total Empathies
        self.loss_weights = None  # Weights for BCE loss to handle Imbalance
        self.word_vectors = {}
        self.mlb = MultiLabelBinarizer()  # Initializing Multilabel Binarizing for Multilabel Encoding

        print(f'Data Shape is: {self.empathy_labeled_data.shape}')
        print("Intial 10 Rows :")
        print(self.empathy_labeled_data.head(10))
        print('\n')
        self.preprocess()
        self.create_dict()

        # message to encoded pattern
        self.empathy_labeled_data['encoded'] = self.empathy_labeled_data['message'].apply(
            lambda x: np.array(self.sentence_encoding(x)))

        # Fill Missing Values with 'idk' and cleans empathy labels
        self.empathy_labeled_data['y_encoded'] = self.empathy_labeled_data['empathy'].apply(
            lambda x: ['idk'] if pd.isnull(x) else [w.strip() for w in x.split(',')])

    def preprocess(self):
        del self.empathy_labeled_data['ignore']  # Ignore Column Contains all NANs values.

        # Processing of text in message column for 'im'. 'im' converts to 'I am' using regex
        self.empathy_labeled_data['message'] = self.empathy_labeled_data['message'].str. \
            replace(r"\bim\b", 'i am', regex=True)

        self.empathy_labeled_data['message'] = self.empathy_labeled_data['message'].apply(lambda x: " ".join(x.split()))

        # Word Lemmatization
        # self.empathy_labeled_data['message'] = self.empathy_labeled_data['message'].apply(
        # lambda x: ' '.join(self.lemmatize_text(x)))
        self.empathy_labeled_data['message_length'] = self.empathy_labeled_data['message'].apply(
            lambda x: len(x.split()))

        # Frequency of the words in corpus
        for index, row in self.empathy_labeled_data.iterrows():
            self.counts.update(self.tokenize(row['message']))

    def sentence_encoding(self, text, N=30):
        tokenized = self.tokenize(text)
        encoded = np.zeros(N, dtype=int)

        # Creates encoding for each message
        enc1 = np.array([self.vocab2index.get(word, self.vocab2index["UNK"]) for word in tokenized])
        length = min(N, len(enc1))  # Constraint on length of message
        encoded[:length] = enc1[:length]  # Zero padding for rest of positions  if length is less than enc1's length
        return encoded, length

    def tokenize(self, text):
        # Tokenize the text
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
        nopunct = regex.sub(" ", text.lower())
        return [token.text for token in tok.tokenizer(nopunct)]

    def describe_counts(self):
        print("Number of words in Corpus: {}".format(len(self.counts)))
        print("Message Avg Length : {}, Message Max Length : {}".format(
            self.empathy_labeled_data['message_length'].mean(),
            max(self.empathy_labeled_data['message_length'])))

    def create_dict(self):
        for word in self.counts:
            # Dictionary for corpus word that maps to unique number
            self.vocab2index[word] = len(self.words)
            # Append the words list
            self.words.append(word)

    def lemmatize_text(self, text):
        # Lemmatization of text
        return [word_lemmatizer.lemmatize(w) for w in word_tokenizer.tokenize(text)]

    def label_binarizer_get_weights(self):
        labels = self.empathy_labeled_data['y_encoded'].values.tolist()

        # Transformation y-labels to Mult-label Binarizer
        y_values = self.mlb.fit_transform(labels)

        # Total Number of Empathies
        self.output_size = len(self.mlb.classes_)
        print(f'Number of Empathies: {self.output_size}, Output Shape is: {y_values.shape}')

        # All empathy-labels in long list
        all_labels_flattened = [w for list_emp in labels for w in list_emp]

        # Counts Frequency of empathy-label in corpus
        count_labels = Counter(all_labels_flattened)

        # Weighted Sampling for Imbalance: higher weight to less frequent Empathy-label
        base = np.zeros(self.output_size)
        weight = np.zeros(self.output_size)

        # Mapping from i to empathy-label
        self.dic = {}
        self.rev_dic = {}
        for i in range(self.output_size):
            base[i] = 1
            empathy_label = self.mlb.inverse_transform(np.array([base]))[0][0]
            self.dic[i] = empathy_label
            self.rev_dic[empathy_label] = i
            # Frequency of Label maps to weight
            weight[i] = count_labels[empathy_label]
            base = np.zeros(len(self.mlb.classes_))

        # Inverses weights for balance sampling
        weight = sum(weight) / weight
        weight = weight / sum(weight)
        self.loss_weights = torch.from_numpy(weight)

        # Final Y-Values for Multi-Label Encoding
        self.empathy_labeled_data['y_encoded_int'] = pd.Series(y_values.tolist())
        print('\n')
        print("First 10 Columns of Data After Data Preprocessing and MultiLabel Encoding:")
        print(self.empathy_labeled_data.head(10))
        print('\n')

        return self.output_size, self.loss_weights

    # Appending numseen_feature in the dataset to their respective encoding
    def add_numseen_feature(self):
        leng = len(self.empathy_labeled_data)
        for i in range(leng):
            self.empathy_labeled_data['encoded'].iloc[i][0] = np.append(self.empathy_labeled_data['encoded'].iloc[i][0],
                                                                        self.empathy_labeled_data['num_seen'].iloc[i])

    def load_glove_vectors(self, glove_file="/media/HDD_2TB.1/glove.6B.50d.txt"):
        """Loading the glove word vectors"""
        with open(glove_file) as f:
            for line in f:
                split = line.split()
                # Fetching GLove vector for Word
                self.word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
        return self.word_vectors

    def get_emb_matrix(self, pretrained, emb_size=50):
        """ Creates embedding matrix from word vectors"""
        vocab_size = len(self.counts) + 2
        vocab_to_idx = {}  # dictionary from word to index
        vocab = ["", "UNK"]
        W = np.zeros((vocab_size, emb_size), dtype="float32")
        W[0] = np.zeros(emb_size, dtype='float32')  # adding a vector for padding
        W[1] = np.random.uniform(-0.25, 0.25, emb_size)  # adding a vector for unknown words
        vocab_to_idx["UNK"] = 1  # "" for 0 , "UNK" for 1, rest of corpus words index starts from 2
        i = 2
        for word in self.counts:
            if word in pretrained:
                W[i] = pretrained[word]  # adding a vector for recognized word
            else:
                W[i] = np.random.uniform(-0.25, 0.25, emb_size)  # adding a vector for unknown words
            vocab_to_idx[word] = i
            vocab.append(word)
            i += 1
        return W, np.array(vocab), vocab_to_idx

    # Retrieves Total Number of Classes
    def get_outputsize(self):
        return self.output_size

    # # Retrieves Weights for BCE-Loss Function
    def get_weights(self):
        return self.loss_weights

    # Retrieves X
    def get_X_data(self):
        return list(self.empathy_labeled_data['encoded'])

    # Retrieves Y
    def get_Y_data(self):
        return list(self.empathy_labeled_data['y_encoded_int'])

    # Retrieves Glove Word Vectors
    def get_word_vec(self):
        return self.word_vectors

    # Method Call for Baseline Modelling, Classifier RandomForest and SVM
    def modelling(self, baseline_classifier, X_train, X_valid, y_train, y_valid):
        # Base Model either Random Forest or SVM
        if baseline_classifier == 'RandomForest':
            clf = MultiOutputClassifier(RandomForestClassifier(max_depth=6, class_weight="balanced", n_jobs=2))
        elif baseline_classifier == 'SVC':
            clf = MultiOutputClassifier(SVC(gamma='auto', class_weight="balanced"))

        clf_train = np.array([row[0] for row in X_train])
        clf_valid = np.array([row[0] for row in X_valid])

        # Fitting the Model and then predict on validation
        clf.fit(clf_train, y_train)
        y_pred = np.array(clf.predict(clf_valid))
        y_valid = np.array(y_valid)

        # Calculating Accuracy and AUC Scores
        return accuracy_score(y_valid.reshape(-1), y_pred.reshape(-1)), \
               roc_auc_score(y_valid.reshape(-1), y_pred.reshape(-1))


class EmpathyDataLoading(Dataset):
    """Data Loader for epochs"""

    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return X-data, Y-value for corresponding item at idx, index
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), torch.from_numpy(
            np.array(self.y[idx]).astype(np.int32)), torch.tensor(self.X[idx][1])
