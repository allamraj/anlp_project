import transformers
from datasets import load_dataset, load_metric
import logging
import json
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs

squad = load_dataset("squad")
mlqa_en_en = load_dataset("mlqa", 'mlqa.en.en')
# mlqa_hi_hi = load_dataset("mlqa", 'mlqa.hi.hi')
model_type = "xlm"
model_name = "xlm-mlm-tlm-xnli15-1024"


# Utility functions for preparing training and test data
# https://www.kaggle.com/code/ejaz22/question-answering-using-simpletransformer/notebook

# Adpated from https://www.kaggle.com/cheongwoongkang/roberta-baseline-starter-simple-postprocessing
# def find_all(input_str,search_str):
#     l1 = []
#     length = len(input_str)
#     index = 0
#     while index < length:
#         i = input_str.find(search_str, index)
#         if i == -1:
#             return l1
#         l1.append(i)
#         index = i + 1
#     return l1


def do_qa_train(train):
    output = []
    for _, qa in enumerate(train):
        # print(qa)
        # break
        context = qa['context']
        qas = []
        question = qa['question']
        qid = qa['id']
        answers = []
        answer = qa['answers']
        # if type(answer) != str or type(context) != str or type(question) != str:
        #     continue

        for i in range(len(answer['text'])):
            answers.append({'answer_start': answer['answer_start'][i], 'text': answer['text'][i].lower()})

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
        output.append({'context': context.lower(), 'qas': qas})

        # print(output)
        # break

    return output


# Prepare testing data in QA-compatible format

# def do_qa_test(test):
#     output = []
#     for line in test:
#         context = line[1]
#         qas = []
#         question = line[-1]
#         qid = line[0]
#         if type(context) != str or type(question) != str:
#             continue
#         answers = []
#         answers.append({'answer_start': 1000000, 'text': '__None__'})
#         qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
#         output.append({'context': context.lower(), 'qas': qas})
#     return output


squad_train = do_qa_train(squad["train"])
squad_val = do_qa_train(squad["validation"])
mlqa_enen_qa = do_qa_train(mlqa_en_en["test"])
mlqa_hi_hi = load_dataset("mlqa", 'mlqa.hi.hi')
mlqa_hihi_qa = do_qa_train(mlqa_en_en["test"])

# Configure the model
model_args = QuestionAnsweringArgs()
model_args.train_batch_size = 16
model_args.evaluate_during_training = False
model_args.n_best_size = 3
model_args.num_train_epochs = 1

### Advanced Methodology
train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "use_cached_eval_features": True,
    "output_dir": f"outputs/{model_type}",
    "best_model_dir": f"outputs/{model_type}/best_model",
    "evaluate_during_training": True,
    "max_seq_length": 128,
    "num_train_epochs": 1,
    "evaluate_during_training_steps": 1000,
    "wandb_project": "Question Answer Application",
    "wandb_kwargs": {"name": model_name},
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "n_best_size": 3,
    # "use_early_stopping": True,
    # "early_stopping_metric": "mcc",
    # "n_gpu": 2,
    # "manual_seed": 4,
    # "use_multiprocessing": False,
    "train_batch_size": 128,
    "eval_batch_size": 64,
    # "config": {
    #     "output_hidden_states": True
    # }
}

model = QuestionAnsweringModel(
    model_type, model_name, args=model_args
)

### Remove output folder
# !rm -rf outputs

# Train the model
model.train_model(squad_train, eval_data=squad_val)

# result, texts = model.eval_model(mlqa_enen_qa)

answers, probabilities = model.predict(mlqa_enen_qa)
answers_hi, probabilities = model.predict(mlqa_hihi_qa)

answers_formatted = {}

for ans in answers:
    answers_formatted[ans['id']] = ans['answer'][0] if ans['answer'] else ''

answers_formatted_hi = {}

for ans in answers_hi:
    answers_formatted_hi[ans['id']] = ans['answer'][0] if ans['answer'] else ''

with open('enen_test.json', 'w', encoding='utf-8') as f:
    json.dump(answers_formatted, f, ensure_ascii=False, indent=4)

with open('hihi_test.json', 'w', encoding='utf-8') as f:
    json.dump(answers_formatted_hi, f, ensure_ascii=False, indent=4)

# !python /content/MLQA/mlqa_evaluation_v1.py\
#    '/content/MLQA_V1/test/test-context-en-question-en.json'\
#    '/content/enen_test.json' \
#    en
# {"exact_match": 38.86108714408973, "f1": 49.33358551461187}
#
# !python /content/MLQA/mlqa_evaluation_v1.py\
#    '/content/MLQA_V1/test/test-context-hi-question-hi.json'\
#    '/content/hihi_test.json' \
#    hi
# {"exact_match": 5.08336722244815, "f1": 7.805575746524095}
