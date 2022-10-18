# Advanced NLP: Project
#
# Authors:
# A Aparajitha <aparajitha.allamraju@research.iiit.ac.in>
# Darshana S <darshana.s@research.iiit.ac.in>
#
# models.py

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base for the pretrained models
    Input
        Language pair: lang1, lang2
        model_name: bert or xlm
        model_type: specific model eg. bert-base-multilingual-cased, xlm-mlm-tlm-xnli15-1024
        train_set: set used for training eg. squad
        val_set:
        eval_set: evaluation with mlqa, mlqa.lang1.lang2
    """
    def __init__(self, lang1, lang2, model_name, model_type, train_set, val_set, eval_set):
        self.lang1 = lang1
        self.lang2 = lang2
        self.model_name = model_name
        self.model_type = model_type
        self.train_set = train_set
        self.val_set = val_set
        self.eval_set = eval_set

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def eval_model(self):
        pass

    @abstractmethod
    def predict_model(self):
        pass
