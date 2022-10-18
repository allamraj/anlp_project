# Advanced NLP: Project
#
# Authors:
# A Aparajitha <aparajitha.allamraju@research.iiit.ac.in>
# Darshana S <darshana.s@research.iiit.ac.in>
#
# clqa_bert.py

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
    """
    def __init__(self, lang1, lang2, model_name, model_type, train_set, val_set, eval_set):
        (super(CLQABERT, self).__init__(lang1, lang2, model_name, model_type, train_set, val_set, eval_set))
        # self.model_checkpoint = "bert-base-multilingual-cased"
        self.batch_size = 32
        self.max_length = 384  # The maximum length of question and context
        self.doc_stride = 128
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pad_on_right = self.tokenizer.padding_side == "right"
        self.tokenized_train = self.get_tokenized_data(self.train_set)
        self.tokenized_val = self.get_tokenized_data(self.val_set)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_config(self.config)
        self.args = self.get_args()
        self.data_collator = default_data_collator
        self.trainer = self.get_trainer()

    def get_trainer(self):
        return Trainer(
            self.model,
            self.args,
            train_dataset=self.tokenized_train["train"],
            eval_dataset=self.tokenized_train["validation"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )

    def get_args(self):
        return TrainingArguments(
            f"{self.model_name}-finetuned-squad",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            push_to_hub=False,
        )

    def get_tokenized_data(self, data):
        """
        Tokenize training data with truncation and padding
        :return: tokenizer
        """
        return self.tokenizer(
            data["question" if self.pad_on_right else "context"],
            data["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

    def prepare_train_features(self):
        # Strip left white space
        self.tokenized_train["question"] = [q.lstrip() for q in self.tokenized_train["question"]]
        sample_mapping = self.tokenized_train.pop("overflow_to_sample_mapping")
        offset_mapping = self.tokenized_train.pop("offset_mapping")

        self.tokenized_train["start_positions"] = []
        self.tokenized_train["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = self.tokenized_train["input_ids"][i]
            sequence_ids = self.tokenized_train.sequence_ids(i)

            sample_index = sample_mapping[i]  # TODO check if needed Handle multiple spans
            answers = self.train_set["answers"][sample_index]
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            span_start = 0
            while sequence_ids[span_start] != (1 if self.pad_on_right else 0):
                span_start += 1

            span_end = len(input_ids) - 1
            while sequence_ids[span_end] != (1 if self.pad_on_right else 0):
                span_end -= 1

            while span_start < len(offsets) and offsets[span_start][0] <= start_char:
                span_start += 1
            self.tokenized_train["start_positions"].append(span_start - 1)

            while offsets[span_end][1] >= end_char:
                span_end -= 1
            self.tokenized_train["end_positions"].append(span_end + 1)

    def prepare_validation_features(self, examples):
        examples["question"] = [q.lstrip() for q in examples["question"]]
        span_map = self.tokenized_val.pop("overflow_to_sample_mapping")
        self.tokenized_val["data_id"] = []

        for i in range(len(self.tokenized_val["input_ids"])):
            sequence_ids = self.tokenized_val.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # Track multiple spans
            span_index = span_map[i]
            self.tokenized_val["data_id"].append(examples["id"][span_index])

            # Set offset mapping to none if not part of context
            self.tokenized_val["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(self.tokenized_val["offset_mapping"][i])
            ]

    def train_model(self):
        self.trainer.train()



validation_features = datasets["validation"].map(
    prepare_validation_features,
    batched=True,
    remove_columns=datasets["validation"].column_names
)

raw_predictions = trainer.predict(validation_features)
validation_features.set_format(type=validation_features.format["type"],
                               columns=list(validation_features.features.keys()))

examples = datasets["validation"]
features = validation_features

example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
features_per_example = collections.defaultdict(list)
for i, feature in enumerate(features):
    features_per_example[example_id_to_index[feature["example_id"]]].append(i)


def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        # if not squad_v2:
        #     predictions[example["id"]] = best_answer["text"]
        # else:
        answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
        predictions[example["id"]] = answer

    return predictions


final_predictions = postprocess_qa_predictions(datasets["validation"], validation_features, raw_predictions.predictions)

metric = load_metric("squad")

formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
metric.compute(predictions=formatted_predictions, references=references)

mlqa = load_dataset('mlqa', 'mlqa.en.en')

validation_features_mlqa = mlqa["validation"].map(
    prepare_validation_features,
    batched=True,
    remove_columns=mlqa["validation"].column_names
)

raw_predictions_mlqa = trainer.predict(validation_features_mlqa)

validation_features_mlqa.set_format(type=validation_features.format["type"],
                                    columns=list(validation_features_mlqa.features.keys()))

# examples = mlqa["validation"]
examples = mlqa["test"]
features = validation_features_mlqa

example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
features_per_example = collections.defaultdict(list)
for i, feature in enumerate(features):
    features_per_example[example_id_to_index[feature["example_id"]]].append(i)

final_predictions_mlqa = postprocess_qa_predictions(mlqa["validation"], validation_features_mlqa,
                                                    raw_predictions_mlqa.predictions)

final_predictions_mlqa = postprocess_qa_predictions(mlqa["test"], validation_features_mlqa,
                                                    raw_predictions_mlqa.predictions)

with open('enen_test.json', 'w', encoding='utf-8') as f:
    json.dump(final_predictions_mlqa, f, ensure_ascii=False, indent=4)

# !python /content/MLQA/mlqa_evaluation_v1.py\
#    '/content/MLQA_V1/test/test-context-en-question-en.json'\
#    '/content/enen_test.json' \
#    en

