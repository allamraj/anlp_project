# Advanced NLP: Project
#
# Authors:
# A Aparajitha <aparajitha.allamraju@research.iiit.ac.in>
# Darshana S <darshana.s@research.iiit.ac.in>
#
# utils.py

import json
import collections
from tqdm.auto import tqdm
import numpy as np
from transformers import AutoTokenizer
from constants import MAX_LEN, DOC_STRIDE

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
pad_on_right = tokenizer.padding_side == "right"


def write_results_to_file(file_path, text):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(text, f, ensure_ascii=False, indent=4)


def get_tokenized_data(data):
    """
    Tokenize training data with truncation and padding
    :return: tokenizer
    """
    return tokenizer(
        data["question" if pad_on_right else "context"],
        data["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=MAX_LEN,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )


def prepare_train_features(data):
    """
    :param data:
    :return:

    Preparing the training data by formatting it to match a Question Answering model format
    """
    # Strip left white space
    data["question"] = [q.lstrip() for q in data["question"]]

    tokenized_train = tokenizer(
        data["question" if pad_on_right else "context"],
        data["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=MAX_LEN,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_train.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_train.pop("offset_mapping")

    tokenized_train["start_positions"] = []
    tokenized_train["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_train["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_train.sequence_ids(i)

        answers = data["answers"][sample_mapping[i]]  # sample_mapping[i] is the sample index
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        span_start = 0
        while sequence_ids[span_start] != (1 if pad_on_right else 0):
            span_start += 1

        span_end = len(input_ids) - 1
        while sequence_ids[span_end] != (1 if pad_on_right else 0):
            span_end -= 1

        # if the answer is out of the span label with CLS index
        if not (offsets[span_start][0] <= start_char and offsets[span_end][1] >= end_char):
            tokenized_train["start_positions"].append(cls_index)
            tokenized_train["end_positions"].append(cls_index)
        else:
            while span_start < len(offsets) and offsets[span_start][0] <= start_char:
                span_start += 1
            tokenized_train["start_positions"].append(span_start - 1)

            while offsets[span_end][1] >= end_char:
                span_end -= 1
            tokenized_train["end_positions"].append(span_end + 1)

    return tokenized_train


def prepare_validation_features(data):
    """
    :param data:
    :return:

    Prepare evaluation data to match validation format for Question Answering
    """
    data["question"] = [q.lstrip() for q in data["question"]]

    tokenized_val = tokenizer(
        data["question" if pad_on_right else "context"],
        data["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=MAX_LEN,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    span_map = tokenized_val.pop("overflow_to_sample_mapping")
    tokenized_val["data_id"] = []

    for i in range(len(tokenized_val["input_ids"])):
        sequence_ids = tokenized_val.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # Track multiple spans
        span_index = span_map[i]
        tokenized_val["data_id"].append(data["id"][span_index])

        # Set offset mapping to none if not part of context
        tokenized_val["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_val["offset_mapping"][i])
        ]

    return tokenized_val


def postprocess_qa_predictions(data, features, raw_predictions, n_best_size=20, max_answer_length=30):
    """

    :param data:
    :param features:
    :param raw_predictions:
    :param n_best_size:
    :param max_answer_length:
    :return:

    Step 0: Create a features dictionary that contain the indices to spans that belong to the same question
    Step 1: Reverse Argsort the start and end logits
    Step 2: Take the n_best_size from both the lists
    Step 3: Go through all possibilities for the n_best_size
    Step 4: Filter out the impossible combinations
    Step 5: Score each combination as sum of probabilities
    Step 6: Answer is the one with the best score
    Step 7: Return empty in case of no valid answer with a score of 0
    """

    all_start_logits, all_end_logits = raw_predictions

    d_to_i = {k: i for i, k in enumerate(data["id"])}  # Map of data_id to index
    # features_dict is a dictionary of lists that contain the indices to spans that belong to the same question
    # Eg. {0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7, 8]}
    features_dict = collections.defaultdict(list)
    for i, feat in enumerate(features):
        features_dict[d_to_i[feat["data_id"]]].append(i)

    predictions = collections.OrderedDict()

    for i, d in enumerate(tqdm(data)):
        valid_answers = []
        context = d["context"]

        for feature_index in features_dict[i]:  # indices of the features associated
            start_logits = all_start_logits[feature_index]  # Start index
            end_logits = all_end_logits[feature_index]  # End index
            offset_mapping = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider the answer if:
                    # if end is less than start span len is greater than max ans len
                    # or start or end are greater len of offset
                    # or start or end don't exist in the mapping
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length or \
                            start_index >= len(offset_mapping) or end_index >= len(offset_mapping) \
                            or offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append({
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char: end_char]
                    }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}  # Step 7: No answer case

        answer = best_answer["text"]
        predictions[d["id"]] = answer

    return predictions
