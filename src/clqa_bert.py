# Advanced NLP: Project
#
# Authors:
# A Aparajitha <aparajitha.allamraju@research.iiit.ac.in>
# Darshana S <darshana.s@research.iiit.ac.in>
#
# clqa_bert.py

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import transformers
import torch
import json
from transformers import AutoConfig, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
from tqdm.auto import tqdm
import numpy as np
import collections
from models import BaseModel


class CLQABERT(BaseModel):
    """
    XLM model for Cross Lingual Question Answering
    Input:
        Language pair -
    """
    def __init__(self, lang1, lang2, model_name, model_type, train_set, val_set, eval_set):
        (super(CLQABERT, self).__init__(lang1, lang2, model_name, model_type, train_set, val_set, eval_set))
        # self.model = self.config_model()
        self.model_checkpoint = "bert-base-multilingual-cased"
        self.batch_size = 32
        self.max_length = 384  # The maximum length of a feature (question and context)
        self.doc_stride = 128
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.pad_on_right = self.tokenizer.padding_side == "right"
        self.tokenized_train = self.get_tokenized_train()

    def get_tokenized_train(self):
        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        return self.tokenizer(
            self.train_set["question" if self.pad_on_right else "context"],
            self.train_set["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

    def prepare_train_features(self):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        self.tokenized_train["question"] = [q.lstrip() for q in self.tokenized_train["question"]]

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = self.tokenized_train.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = self.tokenized_train.pop("offset_mapping")  # list of 20 tuples (beg,end) if context has 20 words

        # Let's label those examples!
        self.tokenized_train["start_positions"] = []
        self.tokenized_train["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = self.tokenized_train["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = self.tokenized_train.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = self.train_set["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                ####### shouldn't happen #######
                self.tokenized_train["start_positions"].append(cls_index)
                self.tokenized_train["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]  ## This is not word index, its char index
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text. find(1)
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text. find(1 from the end)
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    self.tokenized_train["start_positions"].append(cls_index)
                    self.tokenized_train["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    self.tokenized_train["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    self.tokenized_train["end_positions"].append(token_end_index + 1)
