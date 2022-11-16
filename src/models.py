# Advanced NLP: Project
#
# Authors:
# A Aparajitha <aparajitha.allamraju@research.iiit.ac.in>
# Darshana S <darshana.s@research.iiit.ac.in>
#
# models.py

import torch
from abc import ABC, abstractmethod
from transformers import Trainer


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


class CustomTrainer(Trainer):
    """
        Custom Trainer for MML Loss
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        start_labels = inputs.pop("start_positions_topk", None)
        end_labels = inputs.pop("end_positions_topk", None)
        outputs = model(**inputs)
        start_logits = outputs.get("start_logits")
        end_logits = outputs.get("end_logits")
        loss_fct = torch.nn.CrossEntropyLoss()
        loss_tensor = torch.zeros(start_labels.shape)
        for i in range(len(start_labels)):
            start_loss = loss_fct(start_logits, start_labels[:, i])
            end_loss = loss_fct(end_logits, end_labels[:, i])
            loss_tensor[:, i] = start_loss + end_loss

        loss = -torch.sum(torch.log(torch.sum(torch.exp(-loss_tensor - 1e10 * (loss_tensor == 0).float()), 1)))
        return (loss, outputs) if return_outputs else loss
