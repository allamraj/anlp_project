# Advanced NLP: Project
#
# Authors:
# A Aparajitha <aparajitha.allamraju@research.iiit.ac.in>
# Darshana S <darshana.s@research.iiit.ac.in>
#
# clqa_fast.py

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
from models import BaseModel
from utils import prepare_train_features, prepare_validation_features
from constants import *


class FastCLQA(BaseModel):
    """
    Model for Cross Lingual Question Answering
    FastCLQA is for pretrained models that have fast tokenizers
        e.g. BERT, XLM-R
    """
    def __init__(self, model_name, model_type, train_set, eval_set):
        (super(FastCLQA, self).__init__(model_name, model_type))
        # self.model_checkpoint = "bert-base-multilingual-cased"
        self.train_set = train_set  # Train set is the SQuAD with train and validation
        self.eval_set = eval_set
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pad_on_right = self.tokenizer.padding_side == "right"
        self.tokenized_train = self.train_set.map(prepare_train_features,
                                                  batched=True, remove_columns=self.train_set["train"].column_names)
        self.tokenized_val = self.train_set["validation"].map(prepare_validation_features, batched=True,
                                                              remove_columns=self.train_set["validation"].column_names)
        self.tokenized_eval = self.eval_set.map(prepare_validation_features,
                                                batched=True, remove_columns=self.eval_set.column_names)

    def get_trainer(self, model):
        return Trainer(
            model,
            self.get_args(),
            train_dataset=self.tokenized_train["train"],
            eval_dataset=self.tokenized_train["validation"],
            data_collator=default_data_collator,
            tokenizer=self.tokenizer,
        )

    def get_args(self):
        return TrainingArguments(
            f"{self.model_name}-finetuned-squad",
            evaluation_strategy="epoch",
            learning_rate=LR,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=DECAY,
            # push_to_hub=HUB,
        )

    # def get_validation_features(self):
    #     return self.tokenized_eval

    def eval_model(self):
        """
        Predict model on SQuAD validation data
        """
        self.model.predict(self.tokenized_val)

    def predict_model(self):
        """
        Predict the model on the MLQA test data
        :return: predictions
        """
        predictions = self.model.predict(self.tokenized_eval)
        self.tokenized_eval.set_format(type=self.tokenized_eval.format["type"],
                                       columns=list(self.tokenized_eval.features.keys()))
        return predictions

    def train_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.get_trainer(AutoModelForQuestionAnswering.from_pretrained(self.model_name).to(device))
        self.model = model.train()
