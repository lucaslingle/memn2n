import os
import re
import numpy as np

class Sentence:
    def __init__(self, string):
        self.string = string

    def get_tokens(self, drop_punctuation=True):
        if drop_punctuation:
            tokens = re.findall(r"[\w]+", self.string)
            return tokens
        else:
            tokens_punctuation_and_whitespaces = re.split('(\W+)?', self.string)
            tokens_and_punctuation = [x.strip() for x in tokens_punctuation_and_whitespaces if x.strip()]
            return tokens_and_punctuation

    def get_tokens_with_padding(self, max_sentence_len, pad_token, drop_punctuation=True):
        tokens = self.get_tokens(drop_punctuation=drop_punctuation)

        if len(tokens) < max_sentence_len:
            padding_tokens = [pad_token for _ in range(max_sentence_len - len(tokens))]
            tokens.extend(padding_tokens)
            return tokens

        if len(tokens) > max_sentence_len:
            tokens = tokens[0:max_sentence_len]
            return tokens

    def get_padded_int_array(self, vocab_dict, pad_token, unk_token, max_sentence_len, drop_punctuation=True):
        tokens_padded = self.get_tokens_with_padding(max_sentence_len, pad_token, drop_punctuation=drop_punctuation)

        word_to_int = lambda w: vocab_dict[w] if w in vocab_dict else vocab_dict[unk_token]
        word_ints_padded = map(word_to_int, tokens_padded)
        return np.array(word_ints_padded, dtype=np.int32)

class Story:
    def __init__(self):
        # Note:
        #  The bAbI dataset consists of stories.
        #  Each story has one or more sentences, and one or more questions-answer pairs.
        #
        #  Some stories have questions part-way through, then more sentences, and then another question.
        #  Each question's scope is all prior sentences in the story.
        #
        #  We use dictionaries keyed by line number in order to retrieve relevant story sentences for ease-of-use.
        #  When performance is needed, we convert all sentences, questions, and answers to numpy arrays

        self.sentences = []
        self.questions = []
        self.answers = []

        self.sqa_tuples = []

    def sentences_update(self, sentence):
        self.sentences.append(Sentence(sentence))

    def questions_update(self, question):
        self.questions.append(Sentence(question))

    def answers_update(self, answer):
        self.answers.append(Sentence(answer))
        s = self.sentences[:]
        q = self.questions[-1]
        a = self.answers[-1]
        sqa_tuple = (s, q, a)
        self.sqa_tuples.append(sqa_tuple)

    @staticmethod
    def apply_to_sqa_tokens(sqa, f):
        ss = list(map(lambda s: list(map(f, s.get_tokens())), sqa[0]))
        q = list(map(f, sqa[1].get_tokens()))
        a = f(sqa[2].string)

        return (ss, q, a)

class bAbI:

    def __init__(self):
        self.file_partition_types = ['train_or_test', 'task_id']
        self.file_partition_values = {
            'train_or_test': ['train', 'test'],
            'task_id': range(1, 21)
        }

        self.file_prefix_formula = lambda task_id, train_or_test: 'qa{}_'.format(task_id)
        self.file_suffix_formula = lambda task_id, train_or_test: '_{}.txt'.format(train_or_test)

        self.unknown_token = '_UNK'
        self.pad_token = '_PAD'

        # H/t to seominjoon, whose regex for this task I have based mine on. All other code is my own.
        #    https://github.com/seominjoon/memnn-tensorflow/blob/master/read_data.py
        #
        self.s_re = re.compile("^(\d+) ([\w\s\.]+)")
        self.q_re = re.compile("^(\d+) ([\w\s\?]+)\t([\w\,]+)\t([\d\+\s]+)")

        self.vocab_dict = None
        self.max_sentence_len = 0


    def get_fp_for_task(self, data_dir, train_or_test, task_id):
        assert train_or_test in self.file_partition_values['train_or_test']
        assert task_id in self.file_partition_values['task_id']

        prefix = self.file_prefix_formula(task_id, train_or_test)
        suffix = self.file_suffix_formula(task_id, train_or_test)

        matching_files = [fn for fn in os.listdir(data_dir) if fn.startswith(prefix) and fn.endswith(suffix)]
        assert len(matching_files) == 1

        filename = matching_files[0]
        fp = os.path.join(data_dir, filename)

        return fp

    def get_stories(self, fp):
        stories = []

        story = None

        f = open(fp, 'r+')
        for line in f:
            sentence_match = self.s_re.match(line)
            question_match = self.q_re.match(line)

            if question_match:
                story_line_nr, question, answer, supporting_facts = question_match.groups()
                story.questions_update(question)
                story.answers_update(answer)

            elif sentence_match:
                story_line_nr, sentence = sentence_match.groups()
                if int(story_line_nr) == 1:
                    if story is not None:
                        stories.append(story)
                    story = Story()
                story.sentences_update(sentence)

        return stories

    def get_vocab_set_from_sqa_tuples(self, sqa_tuples):
        vocab = set()
        max_sentence_len = 0

        for sqa in sqa_tuples:
            ss = list(map(lambda s: s.get_tokens(), sqa[0]))
            q = sqa[1].get_tokens()
            a = [sqa[2].string]

            s_flat = [token for s in ss for token in s]
            vocab |= set(s_flat)
            vocab |= set(q)
            vocab |= set(a)

            slen = max([len(s) for s in ss])
            qlen = len(q)
            max_sentence_len = max([max_sentence_len, slen, qlen])

        return vocab, max_sentence_len

    def prepare_data_for_single_task(self, data_dir, task_id):
        train_fp = self.get_fp_for_task(data_dir, 'train', task_id)
        test_fp = self.get_fp_for_task(data_dir, 'test', task_id)

        train_stories = self.get_stories(train_fp)
        test_stories = self.get_stories(test_fp)

        train_sqa_tuples = [sqa for story in train_stories for sqa in story.sqa_tuples]
        test_sqa_tuples = [sqa for story in test_stories for sqa in story.sqa_tuples]

        vocab_set, max_sentence_len = self.get_vocab_set_from_sqa_tuples(train_sqa_tuples + test_sqa_tuples)
        vocab_list = list(vocab_set)
        vocab_list.insert(0, self.unknown_token)
        vocab_list.insert(0, self.pad_token)

        vocab_dict = dict({w: i for i, w in enumerate(vocab_list)})

        self.vocab_dict = vocab_dict
        self.max_sentence_len = max_sentence_len

        f = lambda x: self.vocab_dict[x]

        train_sqa_tuples_ints = [Story.apply_to_sqa_tokens(sqa, f) for sqa in train_sqa_tuples]
        test_sqa_tuples_ints = [Story.apply_to_sqa_tokens(sqa, f) for sqa in test_sqa_tuples]

        return train_sqa_tuples_ints, test_sqa_tuples_ints


    def prepare_data_for_joint_tasks(self, data_dir):
        train_sqa_tuples = []
        test_sqa_tuples = []

        for task_id in self.file_partition_values['task_id']:
            train_fp = self.get_fp_for_task(data_dir, 'train', task_id)
            test_fp = self.get_fp_for_task(data_dir, 'test', task_id)

            train_stories = self.get_stories(train_fp)
            test_stories = self.get_stories(test_fp)

            train_sqa_tuples_for_task = [sqa for story in train_stories for sqa in story.sqa_tuples]
            test_sqa_tuples_for_task = [sqa for story in test_stories for sqa in story.sqa_tuples]

            train_sqa_tuples.extend(train_sqa_tuples_for_task)
            test_sqa_tuples.extend(test_sqa_tuples_for_task)

        vocab_set, max_sentence_len = self.get_vocab_set_from_sqa_tuples(train_sqa_tuples + test_sqa_tuples)
        vocab_list = list(vocab_set)
        vocab_list.insert(0, self.unknown_token)
        vocab_list.insert(0, self.pad_token)

        vocab_dict = dict({w: i for i, w in enumerate(vocab_list)})

        self.vocab_dict = vocab_dict
        self.max_sentence_len = max_sentence_len

        f = lambda x: self.vocab_dict[x]

        train_sqa_tuples_ints = [Story.apply_to_sqa_tokens(sqa, f) for sqa in train_sqa_tuples]
        test_sqa_tuples_ints = [Story.apply_to_sqa_tokens(sqa, f) for sqa in test_sqa_tuples]

        return train_sqa_tuples_ints, test_sqa_tuples_ints
















