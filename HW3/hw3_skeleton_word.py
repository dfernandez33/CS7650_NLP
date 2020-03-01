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
            elif n == 1:
                prev_word = grams[index - 1][1]
                grams.append((prev_word, word))
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
        self.vocab = set()
        self.context_counts = {}

    def get_vocab(self):
        ''' Returns the set of words in the vocab '''
        return self.vocab

    def update(self, text: str):
        ''' Updates the model n-grams based on text '''
        grams = ngrams(self.n, text)
        for ngram in grams:
            self.vocab.add(ngram[1])
            if ngram in self.count.keys():
                self.count[ngram] += 1
            else:
                self.count[ngram] = 1
            #  we compute the context counts upfront in order to make the prob calculations faster down the line
            if ngram[0] in self.context_counts.keys():
                self.context_counts[ngram[0]] += 1
            else:
                self.context_counts[ngram[0]] = 1

    def prob(self, context: str, word: str):
        ''' Returns the probability of word appearing after context '''
        # check if the context has been seen before and get context count
        context_seen = False
        context_count = 0
        if context in self.context_counts.keys():
            context_seen = True
            context_count += self.context_counts[context]

        if not context_seen:  # return (1 / vocab_size) if the context has not been seen before
            return 1 / len(self.vocab)

        if self.k > 0:
            context_count += len(self.vocab) * self.k

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
        sorted_vocab = sorted(self.vocab)
        prob_sum = 0
        index = 0
        current_word = None
        while prob_sum <= r:
            current_word = sorted_vocab[index]
            prob_sum += self.prob(context, current_word)
            index += 1

        return current_word

    def random_text(self, length):
        ''' Returns text of the specified word length based on the
            n-grams learned by this model '''
        context = start_pad(self.n).strip().split()  # remove ' ' at end of padding
        if self.n == 0:
            context = ['']  # context is empty for n = 0
        text = []
        for i in range(length):
            str_context = " ".join(context)
            text.append(self.random_word(str_context))
            if self.n > 0:
                context.append(text[i])
                context = context[1:]   # only update context if n > 0

        return " ".join(text)  # remove last space from text

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
        exp = (-1 / len(text.strip().split())) * log_sum  # use length of text without spaces
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
            self.lambdas.append(1.0 / (self.n + 1))  # initialize all lambdas to be equal

    def get_vocab(self):
        return self.vocab

    def update(self, text: str):
        interpolated_grams = []  # this will be a list of ngram lists with N + 1 lists
        count = self.n
        while count >= 0:
            interpolated_grams.append(ngrams(count, text))
            count -= 1
        for ngram in interpolated_grams:  # iterate through every list of ngrams
            for gram in ngram:  # iterated through every ngram in each list
                self.vocab.add(gram[1])
                if gram in self.count.keys():
                    self.count[gram] += 1
                else:
                    self.count[gram] = 1
                #  we compute the context counts upfront in order to make the prob calculations faster down the line
                if gram[0] in self.context_counts.keys():
                    self.context_counts[gram[0]] += 1
                else:
                    self.context_counts[gram[0]] = 1

    def prob(self, context: str, word: str):
        total_prob = self.lambdas[0] * self.prob_helper(context, word)
        context = context.strip().split()  # turn context into list of words
        for i in range(1, len(self.lambdas)):
            context = context[1:]  # remove first word from context each iteration
            total_prob += self.lambdas[i] * self.prob_helper(" ".join(context), word)  # use join to turn list into string

        return total_prob
        # probs_and_context_counts = []
        # context = context.strip().split()  # turn context into list of words
        # for i in range(self.n + 1):
        #     context = context[1:]
        #     probs_and_context_counts.append(self.prob_helper(" ".join(context), word))
        #
        # total_counts = sum([pair[1] for pair in probs_and_context_counts])
        # total_prob = 0
        # for prob, count in probs_and_context_counts:  # set the lambda of each ngram to depending on the frequency of context
        #     curr_lambda = count / total_counts
        #     total_prob += curr_lambda * prob
        # return total_prob

    def prob_helper(self, context, word):
        # check if the context has been seen before and get context count
        context_seen = False
        context_count = 0
        if context in self.context_counts.keys():
            context_seen = True
            context_count += self.context_counts[context]

        if not context_seen:  # return (1 / vocab_size) if the context has not been seen before
            return 1 / len(self.vocab)

        if self.k > 0:
            context_count += len(self.vocab) * self.k

        # count the number of times the specific ngram has been seen
        ngram_count = self.k
        if (context, word) in self.count.keys():
            ngram_count += self.count[(context, word)]

        return ngram_count / context_count

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################


if __name__ == '__main__':
    print("creating model based on train_e.txt using n = 2 and k = .125 with interpolation.")
    m = create_ngram_model(NgramModelWithInterpolation, 'train_e.txt', n=2, k=.03125)
    with open("val_e.txt", encoding='utf-8') as f:
        print(m.random_text(100))
        print("Model perplexity for val_e.txt: " + str(m.perplexity(f.read())))
