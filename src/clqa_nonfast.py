# Advanced NLP: Project
#
# Authors:
# A Aparajitha <aparajitha.allamraju@research.iiit.ac.in>
# Darshana S <darshana.s@research.iiit.ac.in>
#
# clqa_nonfast.py

import torch.cuda
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
from models import BaseModel
from constants import TRAIN_BATCH_SIZE, NUM_EPOCHS, N_BEST_SIZE


class NonFastCLQA(BaseModel):
    """
    Model for Cross Lingual Question Answering
    Model when no fast tokenizer present in HuggingFace
        eg. XLM, MuRIL
    Input:
    QA in simpletransformers requires input data in a specific format
    """
    def __init__(self, model_name, model_type, train_set, eval_set):
        (super(NonFastCLQA, self).__init__(model_name, model_type))
        self.train_set = NonFastCLQA.reformat_data(train_set["train"])
        self.val_set = NonFastCLQA.reformat_data(train_set["validation"])
        self.eval_set = NonFastCLQA.reformat_data(eval_set)

    def get_model(self):
        # Configure the model
        model_args = QuestionAnsweringArgs()
        model_args.train_batch_size = TRAIN_BATCH_SIZE
        model_args.evaluate_during_training = False
        model_args.n_best_size = N_BEST_SIZE
        model_args.num_train_epochs = NUM_EPOCHS

        model = QuestionAnsweringModel(
            self.model_type, self.model_name, args=model_args, use_cuda=torch.cuda.is_available()
        )
        return model

    def train_model(self):
        model = self.get_model()
        self.model = model.train_model(self.train_set, eval_data=self.val_set)

    def eval_model(self):
        result, texts = self.model.eval_model(self.eval_set)
        return result, texts

    def predict_model(self):
        ans, probabilities = self.model.predict(self.eval_set)
        answers = NonFastCLQA.format_answers(ans)
        return answers

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

    # Needed if more than one correct answer is being predicted
    # If n_best is set to more than 1
    @staticmethod
    def format_answers(answers):
        answer_formatted = {}

        for answer in answers:
            answer_formatted[answer['id']] = answer['answer'][0] if answer['answer'] else ''

        return answer_formatted
