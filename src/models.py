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
        model_name: bert or xlm
        model_type: specific model eg. bert-base-multilingual-cased, xlm-mlm-tlm-xnli15-1024
        train_set: set used for training eg. squad
        val_set: validation
        eval_set: evaluation with mlqa, mlqa.lang_q.lang_c
    """
    def __init__(self, model_name, model_type):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None

    # @abstractmethod
    # def get_model(self):
    #     pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def eval_model(self):
        pass

    @abstractmethod
    def predict_model(self):
        pass
