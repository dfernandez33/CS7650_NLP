import math, random
from typing import List, Tuple

################################################################################
# Part 0: Utility Functions
################################################################################


def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~ ' * n


Pair = Tuple[str, str]
Ngrams = List[Pair]


def ngrams(n, text:str) -> Ngrams:
    text = text.strip().split()
    ''' Returns the ngrams of the text as tuples where the first element is
        the n-word sequence (i.e. "I love machine") context and the second is the word '''
    grams = []
    for word, index in zip(text, range(len(text))):
        if index is 0:
            padding = start_pad(n)
            grams.append((padding[:-1], word))  # slice is used to remove last ' ' in padding
        else:
            if n == 0:  # context should always be empty in this case
                grams.append(('', word))
            else:
                prev_word = grams[index - 1][1]
                prev_context = grams[index - 1][0]
                curr_context = prev_context[prev_context.find(' ') + 1:] + ' ' + prev_word
                grams.append((curr_context, word))

    return grams


def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8') as f:
        model.update(f.read())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.count = {}

    def get_vocab(self):
        ''' Returns the set of words in the vocab '''
        vocab = set()
        for key in self.count.keys():
            vocab.add(key[1])
        return vocab

    def update(self, text:str):
        ''' Updates the model n-grams based on text '''
        grams = ngrams(self.n, text)
        for ngram in grams:
            if ngram in self.count.keys():
                self.count[ngram] += 1
            else:
                self.count[ngram] = 1

    def prob(self, context:str, word:str):
        ''' Returns the probability of word appearing after context '''
        # check if the context has been seen before and get context count
        context_seen = False
        context_count = 0
        for key in self.count.keys():
            if context == key[0]:
                context_seen = True
                context_count += self.count[key]

        if not context_seen:  # return (1 / vocab_size) if the context has not been seen before
            return 1 / len(self.get_vocab())

        if self.k > 0:
            context_count += len(self.get_vocab()) * self.k

        # count the number of times the specific ngram has been seen
        ngram_count = self.k
        if (context, word) in self.count.keys():
            ngram_count += self.count[(context, word)]

        return ngram_count / context_count

    def random_word(self, context):
        ''' Returns a random word based on the given context and the
            n-grams learned by this model '''
#         random.seed(1)
        r = random.random()
        vocab = sorted(self.get_vocab())
        prob_sum = 0
        index = 0
        current_word = None
        while prob_sum <= r:
            current_word = vocab[index]
            prob_sum += self.prob(context, current_word)
            index += 1

        return current_word

    def random_text(self, length):
        ''' Returns text of the specified word length based on the
            n-grams learned by this model '''
        context = start_pad(self.n)
        text = ""
        for i in range(length):
            text += self.random_word(context) + ' '
            if self.n > 0:
                context = context[1:] + text[i]

        return text[:-1]  # remove last space from text

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        text_ngrams = ngrams(self.n, text)
        log_sum = 0
        for n_gram in text_ngrams:
            prob = self.prob(n_gram[0], n_gram[1])
            if prob == 0:
                return float('inf')
            log_sum += math.log(prob, 2)
        exp = (-1 / len(text)) * log_sum
        return 2 ** exp

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        super().__init__(n, k)
        self.lambdas = []
        for i in range(self.n + 1):
            self.lambdas.append(1.0 / (self.n + 1))

    def get_vocab(self):
        vocab = set()
        for key in self.count.keys():
            vocab.add(key[1])
        return vocab

    def update(self, text:str):
        interpolated_grams = []  # this will be a list of ngram lists with N + 1 lists
        count = self.n
        while count >= 0:
            interpolated_grams.append(ngrams(count, text))
            count -= 1
        for ngram in interpolated_grams:  # iterate through every list of ngrams
            for gram in ngram:  # iterated through every ngram in each list
                if gram in self.count.keys():
                    self.count[gram] += 1
                else:
                    self.count[gram] = 1

    def prob(self, context:str, word:str):
        total_prob = self.lambdas[0] * self.prob_helper(context, word)
        context = context.strip().split()  # turn context into list of words
        print(context)
        for i in range(1, len(self.lambdas)):
            context = context[1:]  # remove first word from context each iteration
            print(context)
            print(" ".join(context))
            total_prob += self.lambdas[i] * self.prob_helper(" ".join(context), word)  # use join to turn list into string

        return total_prob

    def prob_helper(self, context, word):
        # check if the context has been seen before and get context count
        context_seen = False
        context_count = 0
        for key in self.count.keys():
            if context == key[0]:
                context_seen = True
                context_count += self.count[key]

        if not context_seen:  # return (1 / vocab_size) if the context has not been seen before
            return 1 / len(self.get_vocab())

        if self.k > 0:
            context_count += len(self.get_vocab()) * self.k

        # count the number of times the specific ngram has been seen
        ngram_count = self.k
        if (context, word) in self.count.keys():
            ngram_count += self.count[(context, word)]

        return ngram_count / context_count

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass