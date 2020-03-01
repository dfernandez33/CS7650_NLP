import math, random

################################################################################
# Part 0: Utility Functions
################################################################################


def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n


def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    grams = []
    for char, index in zip(text, range(len(text))):
        if index is 0:
            grams.append((start_pad(n), char))
        else:
            if n == 0:  # context should always be empty in this case
                grams.append(('', char))
            else:
                prev_char = grams[index - 1][1]
                prev_context = grams[index - 1][0]
                curr_context = prev_context[1:len(prev_context)] + prev_char
                grams.append((curr_context, char))

    return grams


def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
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
        self.model_ngrams = {}
        self.vocab = set()
        self.context_counts = {}

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        grams = ngrams(self.n, text)
        for ngram in grams:
            self.vocab.add(ngram[1])
            if ngram in self.model_ngrams.keys():
                self.model_ngrams[ngram] += 1
            else:
                self.model_ngrams[ngram] = 1
            #  we compute the context counts upfront in order to make the prob calculations faster down the line
            if ngram[0] in self.context_counts.keys():
                self.context_counts[ngram[0]] += 1
            else:
                self.context_counts[ngram[0]] = 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        # check if the context has been seen before and get context count
        context_seen = False
        context_count = 0
        if context in self.context_counts.keys():
            context_seen = True
            context_count += self.context_counts[context]

        if not context_seen:  # return (1 / vocab_size) if the context has not been seen before
            return 1 / len(self.get_vocab())

        if self.k > 0:
            context_count += len(self.get_vocab()) * self.k

        # count the number of times the specific ngram has been seen
        ngram_count = self.k
        if (context, char) in self.model_ngrams.keys():
            ngram_count += self.model_ngrams[(context, char)]

        return ngram_count / context_count

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
#         random.seed(1)
        r = random.random()
        vocab = sorted(self.get_vocab())
        prob_sum = 0
        index = 0
        current_char = None
        while prob_sum <= r:
            current_char = vocab[index]
            prob_sum += self.prob(context, current_char)
            index += 1

        return current_char

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        context = start_pad(self.n)
        text = ""
        for i in range(length):
            text += self.random_char(context)
            if self.n > 0:
                context = context[1:] + text[i]

        return text

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
            self.lambdas.append(1.0 / (self.n + 1))  # initialize all lambdas to be equal

    def get_vocab(self):
        return self.vocab

    def update(self, text):
        interpolated_grams = []  # this will be a list of ngram lists with N + 1 lists
        count = self.n
        while count >= 0:
            interpolated_grams.append(ngrams(count, text))
            count -= 1
        for ngram in interpolated_grams:  # iterate through every list of ngrams
            for gram in ngram:  # iterated through every ngram in each list
                self.vocab.add(gram[1])
                if gram in self.model_ngrams.keys():
                    self.model_ngrams[gram] += 1
                else:
                    self.model_ngrams[gram] = 1
                #  we compute the context counts upfront in order to make the prob calculations faster down the line
                if gram[0] in self.context_counts.keys():
                    self.context_counts[gram[0]] += 1
                else:
                    self.context_counts[gram[0]] = 1

    def prob(self, context, char):
        # probs_and_context_counts = []
        # for i in range(self.n + 1):
        #     context = context[1:]
        #     probs_and_context_counts.append(self.prob_helper(context, char))
        #
        # total_counts = sum([pair[1] for pair in probs_and_context_counts])
        # total_prob = 0
        # for prob, count in probs_and_context_counts:  # set the lambda of each ngram to depending on the frequency of context
        #     curr_lambda = count / total_counts
        #     total_prob += curr_lambda * prob
        # return total_prob
        total_prob = self.lambdas[0] * self.prob_helper(context, char)
        for i in range(1, len(self.lambdas)):
            context = context[1:]  # remove first word from context each iteration
            total_prob += self.lambdas[i] * self.prob_helper(context, char)  # use join to turn list into string

        return total_prob

    def prob_helper(self, context, char):
        # check if the context has been seen before and get context count
        context_seen = False
        context_count = 0
        if context in self.context_counts.keys():
            context_seen = True
            context_count += self.context_counts[context]

        if not context_seen:  # return (1 / vocab_size) if the context has not been seen before
            return 1 / len(self.get_vocab())

        if self.k > 0:
            context_count += len(self.get_vocab()) * self.k

        # count the number of times the specific ngram has been seen
        ngram_count = self.k
        if (context, char) in self.model_ngrams.keys():
            ngram_count += self.model_ngrams[(context, char)]

        return ngram_count / context_count

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    print("creating model based on shakespeare_inputs.txt using n = 4 and k = .05 with interpolation.")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', n=4, k=.05)
    with open("shakespeare_sonnets.txt", encoding='utf-8') as f:
        print("Model perplexity for shakespeare_sonnexts.txt: " + str(m.perplexity(f.read())))