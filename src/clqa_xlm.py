# Advanced NLP: Project
#
# Authors:
# A Aparajitha <aparajitha.allamraju@research.iiit.ac.in>
# Darshana S <darshana.s@research.iiit.ac.in>
#
# clqa_xlm.py

import transformers
from datasets import load_dataset, load_metric
import logging
import json
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
from models import BaseModel


class CLQAXLM(BaseModel):
    """
    XLM model for Cross Lingual Question Answering
    Input:
        Language pair -
    """
    def __init__(self, lang1, lang2, model_name, model_type, train_set, val_set, eval_set):
        (super(CLQAXLM, self).__init__(lang1, lang2, model_name, model_type, train_set, val_set, eval_set))
        self.train_set = CLQAXLM.reformat_data(self.train_set)
        self.val_set = CLQAXLM.reformat_data(self.val_set)
        self.eval_set = CLQAXLM.reformat_data(self.eval_set)
        self.model = self.config_model()

    def config_model(self):
        # Configure the model
        model_args = QuestionAnsweringArgs()
        model_args.train_batch_size = 16
        model_args.evaluate_during_training = False
        model_args.n_best_size = 3
        model_args.num_train_epochs = 1

        model = QuestionAnsweringModel(
            self.model_type, self.model_name, args=model_args
        )
        return model

    def train_model(self):
        self.model.train_model(self.train_set, eval_data=self.val_set)

    def eval_model(self):
        result, texts = self.model.eval_model(self.eval_set)

    def predict_model(self):
        ans, probabilities = self.model.predict(self.eval_set)
        answers = CLQAXLM.format_answers(ans)
        return answers, probabilities

    @staticmethod
    def reformat_data(dataset):
        """
        Simpletransformer requires data as

        train_data = [
        {
            "context": "Context",
            "qas": [
                {
                    "id": "00001",
                    "is_impossible": False,
                    "question": "Q",
                    "answers": [ {
                            "text": "text",
                            "answer_start": 1, }],
                }],}, ... ]
        """
        output = []
        for _, qa in enumerate(dataset):
            qas = []
            answers = []
            answer = qa['answers']

            for i in range(len(answer['text'])):
                answers.append({'answer_start': answer['answer_start'][i], 'text': answer['text'][i].lower()})

            qas.append({'question': qa['question'], 'id': qa['id'], 'is_impossible': False, 'answers': answers})
            output.append({'context': qa['context'].lower(), 'qas': qas})

        return output

    @staticmethod
    def format_answers(answers):
        answer_formatted = {}

        for answer in answers:
            answer_formatted[answer['id']] = answer['answer'][0] if answer['answer'] else ''

        return answer_formatted

    # !python /content/MLQA/mlqa_evaluation_v1.py\
    #    '/content/MLQA_V1/test/test-context-en-question-en.json'\
    #    '/content/enen_test.json' \
    #    en
    # {"exact_match": 38.86108714408973, "f1": 49.33358551461187}
    #
    # !python /content/MLQA/mlqa_evaluation_v1.py\
    #    '/content/MLQA_V1/test/test-context-hi-question-hi.json'\
    #    '/content/hihi_test.json' \
    #    hi
    # {"exact_match": 5.08336722244815, "f1": 7.805575746524095}
