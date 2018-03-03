import os
import re
import numpy as np
import pickle
import math
import errno
import collections

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

    @staticmethod
    def pad_tokens(tokens, max_sentence_len, pad_token):

        if len(tokens) < max_sentence_len:
            padding_tokens = [pad_token for _ in range(max_sentence_len - len(tokens))]
            tokens.extend(padding_tokens)
            return tokens

        if len(tokens) > max_sentence_len:
            tokens = tokens[0:max_sentence_len]
            return tokens

    @staticmethod
    def padded_int_array(sentence_ints, pad_id, max_sentence_len):
        sentence_ints_array = np.array(sentence_ints, dtype=np.int32)
        padding_array = pad_id * np.ones((max_sentence_len - len(sentence_ints)), dtype=np.int32)

        return np.concatenate([sentence_ints_array, padding_array])


_SQATuple = collections.namedtuple("SQATuple", ("story_task_id", "context_sentences", "question", "answer"))

class SQATuple(_SQATuple):
  """
    Stores the context sentences of a story up to a question, the question itself, and the answer.
  """
  __slots__ = ()


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

        self.story_task_id = None

        self.sentences = []
        self.questions = []
        self.answers = []

        self.sqa_tuples = []

    def set_story_task_id(self, task_id):
        if self.story_task_id is None:
            self.story_task_id = task_id
        else:
            raise AttributeError(errno.ENOTSUP, os.strerror(errno.ENOTSUP), "task id for story is immutable")

    def sentences_update(self, sentence):
        self.sentences.append(Sentence(sentence))

    def questions_update(self, question):
        self.questions.append(Sentence(question))

    def answers_update(self, answer):
        self.answers.append(Sentence(answer))

        task_id = self.story_task_id
        s = self.sentences[:]
        q = self.questions[-1]
        a = self.answers[-1]

        sqa_tuple = SQATuple(task_id, s, q, a)
        self.sqa_tuples.append(sqa_tuple)

    @staticmethod
    def apply_to_sqa_tokens(sqa, f):
        task_id = sqa.story_task_id

        ss = list(map(lambda sentence: [f(token) for token in sentence.get_tokens()], sqa.context_sentences))
        q = [f(token) for token in sqa.question.get_tokens()]
        a = f(sqa.answer.string)

        return SQATuple(task_id, ss, q, a)


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

    def get_stories(self, data_dir, train_or_test, task_id):
        fp = self.get_fp_for_task(data_dir, train_or_test, task_id)
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
                    story.set_story_task_id(task_id)
                story.sentences_update(sentence)

        stories.append(story)

        return stories

    def compute_max_sentence_len_from_sqa_tuples(self, sqa_tuples):
        max_sentence_len = 0

        for sqa in sqa_tuples:
            ss_lens = list(map(lambda sentence: len(sentence.get_tokens()), sqa.context_sentences))
            q_len = len(sqa.question.get_tokens())

            ss_max_len = max(ss_lens)

            max_sentence_len = max([max_sentence_len, ss_max_len, q_len])

        return max_sentence_len

    def compute_vocab_set_from_sqa_tuples(self, sqa_tuples):
        vocab = set()

        for sqa in sqa_tuples:
            ss_tokens = list(map(lambda sentence: sentence.get_tokens(), sqa.context_sentences))
            q_tokens = sqa.question.get_tokens()
            a_token = sqa.answer.string

            ss_tokens_flat = [token for sentence_tokens in ss_tokens for token in sentence_tokens]
            vocab |= set(ss_tokens_flat)
            vocab |= set(q_tokens)
            vocab.add(a_token)

        return vocab

    def compute_vocab_dict_from_sqa_tuples(self, sqa_tuples):
        vocab_set = self.compute_vocab_set_from_sqa_tuples(sqa_tuples)
        vocab_list = sorted(list(vocab_set))
        vocab_list.insert(0, self.unknown_token)
        vocab_list.insert(0, self.pad_token)

        vocab_dict = dict({w: i for i, w in enumerate(vocab_list)})
        return vocab_dict

    def save_vocab_dict_to_file(self, data, fp):
        with open(fp, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("[*] Successfully saved vocab dictionary to file {}".format(fp))
            return

    def load_vocab_dict_from_file(self, fp):
        with open(fp, 'rb') as handle:
            vocab_dict = pickle.load(handle)
            print("[*] Successfully loaded vocab dictionary from file {}".format(fp))
            return vocab_dict

    def save_max_sentence_len_to_file(self, data, fp):
        with open(fp, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("[*] Successfully saved max_sentence_len to file {}".format(fp))
            return

    def load_max_sentence_len_from_file(self, fp):
        with open(fp, 'rb') as handle:
            max_sentence_len = pickle.load(handle)
            print("[*] Successfully loaded max_sentence_len from file {}".format(fp))
            return max_sentence_len

    def _prepare_data_for_task_ids(self, data_dir, task_ids, validation_frac, vocab_dict=None, max_sentence_len=None):
        train_sqa_tuples_for_all_tasks = []
        validation_sqa_tuples_for_all_tasks = []
        test_sqa_tuples_for_all_tasks = []

        for task_id in task_ids:
            train_stories_for_task = self.get_stories(data_dir, 'train', task_id)
            test_stories_for_task = self.get_stories(data_dir, 'train', task_id)

            # each task has stories, each story has SQA tuples
            # SQA tuples consist of
            #   - the story's cumulative context up to the question,
            #   - the question,
            #   - the answer

            train_sqa_tuples_for_task = [sqa for story in train_stories_for_task for sqa in story.sqa_tuples]
            test_sqa_tuples_for_task = [sqa for story in test_stories_for_task for sqa in story.sqa_tuples]

            # Our train/val split will be stratified by task.
            #
            # However, the split will be performed over the list of SQA tuples, not the list of stories.
            # Thus, there may be questions from any given story that are omitted from training set,
            # but we aren't omitting entire stories from the training set
            #
            # Note that, during training, the list of SQA tuples may have the SQA tuples from a given story
            # presented out of order,
            # e.g., we might train on an SQA tuple   ([S1,S2,S3], Q2, A2)
            #       before training on the SQA tuple ([S1,S2], Q1, A1)    from that story.
            #
            # However, the order of the sentences contained WITHIN any given SQA tuple will remain intact.
            # I.e., S1 really is the first sentence, S2 really is the second sentence, etc.
            #
            # This is because the behavior of np.random.shuffle does not change the contents of each element of the list

            np.random.shuffle(train_sqa_tuples_for_task)

            validation_frac_size = math.floor(validation_frac * len(train_sqa_tuples_for_task))
            split_idx = len(train_sqa_tuples_for_task) - validation_frac_size

            _tmp = train_sqa_tuples_for_task[:]
            train_sqa_tuples_for_task = _tmp[0:split_idx]
            validation_sqa_tuples_for_task = _tmp[split_idx:]

            train_sqa_tuples_for_all_tasks.extend(train_sqa_tuples_for_task)
            validation_sqa_tuples_for_all_tasks.extend(validation_sqa_tuples_for_task)
            test_sqa_tuples_for_all_tasks.extend(test_sqa_tuples_for_task)

        # once we are done with all tasks, shuffle the training set again.
        np.random.shuffle(train_sqa_tuples_for_all_tasks)

        sqa_tuples_for_vocab = []
        sqa_tuples_for_vocab.extend(train_sqa_tuples_for_all_tasks)
        sqa_tuples_for_vocab.extend(validation_sqa_tuples_for_all_tasks)
        sqa_tuples_for_vocab.extend(test_sqa_tuples_for_all_tasks)

        sqa_tuples_for_max_sentence_len = []
        sqa_tuples_for_max_sentence_len.extend(train_sqa_tuples_for_all_tasks)
        sqa_tuples_for_max_sentence_len.extend(validation_sqa_tuples_for_all_tasks)
        sqa_tuples_for_max_sentence_len.extend(test_sqa_tuples_for_all_tasks)

        if vocab_dict is None:
            vocab_dict = self.compute_vocab_dict_from_sqa_tuples(sqa_tuples_for_vocab)

        if max_sentence_len is None:
            max_sentence_len = self.compute_max_sentence_len_from_sqa_tuples(sqa_tuples_for_max_sentence_len)

        self.vocab_dict = vocab_dict
        self.max_sentence_len = max_sentence_len

        f = lambda x: self.vocab_dict[x] if x in self.vocab_dict else self.vocab_dict[self.unknown_token]

        train_sqa_tuples_ints = [Story.apply_to_sqa_tokens(sqa, f) for sqa in train_sqa_tuples_for_all_tasks]
        validation_sqa_tuples_ints = [Story.apply_to_sqa_tokens(sqa, f) for sqa in validation_sqa_tuples_for_all_tasks]
        test_sqa_tuples_ints = [Story.apply_to_sqa_tokens(sqa, f) for sqa in test_sqa_tuples_for_all_tasks]

        return train_sqa_tuples_ints, validation_sqa_tuples_ints, test_sqa_tuples_ints

    def prepare_data_for_single_task(self, data_dir, task_id, validation_frac, vocab_dict=None, max_sentence_len=None):
        task_ids = [task_id]
        tr, va, te = self._prepare_data_for_task_ids(data_dir, task_ids, validation_frac, vocab_dict, max_sentence_len)
        return tr, va, te

    def prepare_data_for_joint_tasks(self, data_dir, validation_frac, vocab_dict=None, max_sentence_len=None):
        task_ids = self.file_partition_values['task_id']
        tr, va, te = self._prepare_data_for_task_ids(data_dir, task_ids, validation_frac, vocab_dict, max_sentence_len)
        return tr, va, te

    @staticmethod
    def standardize_features(sqa, max_sentence_length_J, number_of_memories_M, pad_id, intersperse_empty_memories=False):
        sentences_ints = sqa.context_sentences[:]
        question_ints = sqa.question
        answer_int = sqa.answer

        # Per Section 4.2:
        #    "The capacity of memory is restricted to the most recent 50 sentences."
        #
        # If the memory network can store M memories, we store only the M most recent sentences.
        #
        nr_sentences = len(sentences_ints)
        start_idx = max(0, (nr_sentences - number_of_memories_M))
        end_idx = nr_sentences
        sentences_ints = sentences_ints[start_idx:end_idx]

        Jpadded_sentences_ints_list = list(map(
            lambda s: Sentence.padded_int_array(s, pad_id=pad_id, max_sentence_len=max_sentence_length_J),
            sentences_ints))

        # Per Section 4.1:
        #     "Note that sentences are indexed in reverse order, reflecting their relative distance from the question
        #      so that x1 is the last sentence of the story."
        #
        Jpadded_sentences_ints_list = Jpadded_sentences_ints_list[::-1]

        Jpadded_question_ints = Sentence.padded_int_array(question_ints, pad_id=pad_id, max_sentence_len=max_sentence_length_J)

        sentences_2d_array = pad_id * np.ones((number_of_memories_M, max_sentence_length_J), dtype=np.int32)
        empty_memory_timeword_id = number_of_memories_M
        timeword_array = empty_memory_timeword_id * np.ones(number_of_memories_M, dtype=np.int32)

        if intersperse_empty_memories:
            nr_sentences = len(Jpadded_sentences_ints_list)

            # This implementation is based on my understanding of the paper and the official implementation.
            # The details in the paper were ambiguous, and this is my attempt to understand it, and the matlab code from Facebook.
            #
            # Other than the official matlab implementation, I have not found anyone else who has implemented random noise,
            # so I don't have any other python code to check this against.
            #
            # For matlab code, see:
            # https://github.com/facebook/MemNN/blob/master/MemN2N-babi-matlab/train.m#L31

            extra_spaces = max(0, number_of_memories_M - nr_sentences)
            max_nr_empty_memories_to_intersperse = min(extra_spaces, int(math.ceil(0.10 * nr_sentences)))
            nr_empty_memories_to_intersperse = 0

            if max_nr_empty_memories_to_intersperse > 0:
                nr_empty_memories_to_intersperse = np.random.randint(low=0, high=max_nr_empty_memories_to_intersperse)

            permutation = np.random.permutation(nr_sentences + nr_empty_memories_to_intersperse)
            set_of_idxs_for_nonempty_memories = set(permutation[0:nr_sentences])
            target_idxs_for_nonempty_memories = sorted(list(set_of_idxs_for_nonempty_memories))

            for i in range(0,nr_sentences):
                target_idx_for_nonempty_memory_i = target_idxs_for_nonempty_memories[i]
                Jpadded_sentence_ints = Jpadded_sentences_ints_list[i]
                sentences_2d_array[target_idx_for_nonempty_memory_i,:] = np.array(Jpadded_sentence_ints)
                timeword_array[target_idx_for_nonempty_memory_i] = target_idx_for_nonempty_memory_i

        else:
            nr_sentences = len(Jpadded_sentences_ints_list)
            sentences_2d_array[0:nr_sentences,:] = np.array(Jpadded_sentences_ints_list, dtype=np.int32)
            timeword_array[0:nr_sentences] = np.array(range(0,nr_sentences), dtype=np.int32)

        return sentences_2d_array, timeword_array, Jpadded_question_ints, answer_int

