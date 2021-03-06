from nltk.tokenize import regexp_tokenize
import numpy as np
import heapq
import re

# Here is a default pattern for tokenization, you can substitue it with yours
default_pattern =  r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """

def tokenize(text, pattern = default_pattern):
    """Tokenize sentence with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    text = text.lower()
    # Uncomment the lines below to remove special characters, punctuation, and URLS from the text.
    # text = re.sub(r'(https|http)?:\/\/.*[\r\n]*', '', text)
    # text = re.sub(r'([^\s\w]|_)+', '', text)
    return regexp_tokenize(text, pattern)


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass



class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {}
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.unigram))
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                feature[self.unigram[text[i].lower()]] += 1
        
        return feature
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)
            

class BigramFeature(FeatureExtractor):
    """Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self):
        # Add your code here!
        self.bigrams = {}

    def fit(self, text_set):
        bigram_frequenies = {}
        for document in text_set:
            document_bigrams = self.generate_bigrams(document)
            for bigram in document_bigrams:
                if bigram not in bigram_frequenies.keys():
                    bigram_frequenies[bigram] = 1
                else:
                    bigram_frequenies[bigram] += 1

        most_frequent_bigrams = heapq.nlargest(5000, bigram_frequenies, key=bigram_frequenies.get)
        index = 0
        for bigram in most_frequent_bigrams:
            self.bigrams[bigram] = index
            index += 1

    def transform(self, text):
        # Add your code here!
        feature = np.zeros(len(self.bigrams))
        bigrams = zip(*[text[i:] for i in range(2)])
        for bigram in bigrams:
            if bigram in self.bigrams:
                feature[self.bigrams[bigram]] += 1

        return feature

    def transform_list(self, text_set):
        # Add your code here!
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))

        return np.array(features)

    def generate_bigrams(self, text):
        return zip(*[text[i:] for i in range(2)])

class CustomFeature(FeatureExtractor):
    """customized feature extractor, such as TF-IDF
    """
    def __init__(self):
        # Add your code here!
        self.bigrams = {}
        self.word_idf_values = {}
        self.most_frequent_bigrams = None

    def fit(self, text_set):
        # Add your code here!

        # create bigram index dictionary from the top 5,000 most frequent bigrams
        bigram_frequenies = {}
        for document in text_set:
            document_bigrams = self.generate_bigrams(document)
            for bigram in document_bigrams:
                if bigram not in bigram_frequenies.keys():
                    bigram_frequenies[bigram] = 1
                else:
                    bigram_frequenies[bigram] += 1

        self.most_frequent_bigrams = heapq.nlargest(5000, bigram_frequenies, key=bigram_frequenies.get)
        index = 0
        for bigram in self.most_frequent_bigrams:
            self.bigrams[bigram] = index
            index += 1

        # calculate inverse document frequency values for each word
        for bigram in self.most_frequent_bigrams:
            document_count = 0
            for document in text_set:
                document_bigrams = self.generate_bigrams(document)
                if bigram in document_bigrams:
                    document_count += 1

            self.word_idf_values[bigram] = np.log(len(text_set) / (1 + document_count))


    def transform(self, text):
        # Add your code here!
        document_bigrams = list(self.generate_bigrams(text))

        term_counts = {}
        for bigram in document_bigrams:
            if bigram in term_counts.keys():
                term_counts[bigram] += 1
            else:
                term_counts[bigram] = 1

        term_frequencies = {}
        for bigram in document_bigrams:
            term_frequencies[bigram] = term_counts[bigram] / len(document_bigrams)

        features = np.zeros(len(self.most_frequent_bigrams))
        for bigram in document_bigrams:
            if bigram in self.most_frequent_bigrams:
                features[self.bigrams[bigram]] = term_frequencies[bigram] * self.word_idf_values[bigram]

        return features


    def transform_list(self, text_set):
        # Add your code here!
        features = []
        for document in text_set:
            features.append(self.transform(document))

        return np.array(features)

    def generate_bigrams(self, text):
        return zip(*[text[i:] for i in range(2)])
