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


def reformat_data(data):
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
    for _, qa in enumerate(data):
        qas = []
        answers = []
        answer = qa['answers']

        for i in range(len(answer['text'])):
            answers.append({'answer_start': answer['answer_start'][i], 'text': answer['text'][i].lower()})

        qas.append({'question': qa['question'], 'id': qa['id'], 'is_impossible': False, 'answers': answers})
        output.append({'context': qa['context'].lower(), 'qas': qas})

    return output


def config_model(model_type, model_name):
    # Configure the model
    model_args = QuestionAnsweringArgs()
    model_args.train_batch_size = 16
    model_args.evaluate_during_training = False
    model_args.n_best_size = 3
    model_args.num_train_epochs = 1

    model = QuestionAnsweringModel(
        model_type, model_name, args=model_args
    )
    return model
