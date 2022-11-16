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
    :param data: SQuAD training data
    :return: Tokenized training data

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
    :param data: MLQA test data OR SQuAD validation data
    :return: tokenized test/validation data

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

    :param data: predictions from the model
    :param features: clqa.tokenized_eval or the validation features
    :param raw_predictions: predictions.predictions
    :param n_best_size:
    :param max_answer_length:
    :return: Processed predictions

    Step 0: Create a features dictionary that contain the indices to spans that belong to the same question
    Step 1: Reverse argsort the start and end logits
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
                    # end is less than start span len is greater than max ans len
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


def prepare_train_features_topk(data):
    """
    Prepare train features for top 20 answers
    :param data:
    :return:
    """
    data["question"] = [q.lstrip() for q in data["question"]]

    tokenized_train = tokenizer(
        data["question" if pad_on_right else "context"],
        data["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_train.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_train.pop("offset_mapping")

    tokenized_train["start_positions"] = []
    tokenized_train["end_positions"] = []
    tokenized_train["start_positions_topk"] = []
    tokenized_train["end_positions_topk"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_train["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_train.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = data["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_train["start_positions_topk"].append([cls_index])
            tokenized_train["end_positions_topk"].append([cls_index])
            tokenized_train["start_positions"].append([cls_index])
            tokenized_train["end_positions"].append([cls_index])
        else:
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_train["start_positions"].append(cls_index)
                tokenized_train["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_train["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_train["end_positions"].append(token_end_index + 1)

            start_pos = []
            end_pos = []
            for start_char, end_char in zip(data["answer_start"][sample_index],
                                            data["answer_end"][sample_index]):

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    start_pos.append(cls_index)
                    end_pos.append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_pos.append(token_start_index - 1)  # list of tokens

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_pos.append(token_end_index + 1)  # list of tokens

            tokenized_train["start_positions_topk"].append(start_pos)
            tokenized_train["end_positions_topk"].append(end_pos)

    return tokenized_train


def postprocess_qa_predictions_topk(data, features, raw_predictions, n_best_size=50, max_answer_length=30):
    """
    :param data: predictions from the model
    :param features: clqa.tokenized_eval or the validation features
    :param raw_predictions: predictions.predictions
    :param n_best_size:
    :param max_answer_length:
    :return: Processed predictions
    """
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(data["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = []

    for example_index, example in enumerate(tqdm(data)):
        feature_indices = features_per_example[example_index]

        valid_answers = []
        start_tokens = []
        end_tokens = []
        pred = example

        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char],
                            "start": start_char,
                            "end": end_char
                        }
                    )

        if len(valid_answers) > 20:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0:20]
        else:
            best_answer = valid_answers
            print("NOOOOOOO {}".format(example["id"]))
        answers = []
        for ans in best_answer:
            answers.append(ans["text"])
            start_tokens.append(ans["start"])
            end_tokens.append(ans["end"])

        start = example["answers"]["answer_start"][0]
        end = start + len(example["answers"]["text"][0])
        if (start, end) not in set(zip(start_tokens, end_tokens)):
            start_tokens[-1] = start
            end_tokens[-1] = end
        pred["text"] = answers
        pred["answer_start"] = start_tokens
        pred["answer_end"] = end_tokens

        predictions.append(pred)

    return predictions
