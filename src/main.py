# Advanced NLP: Project
#
# Authors:
# A Aparajitha <aparajitha.allamraju@research.iiit.ac.in>
# Darshana S <darshana.s@research.iiit.ac.in>
#
# main.py

import argparse
import sys
from datasets import load_dataset
from clqa_nonfast import NonFastCLQA
from clqa_fast import FastCLQA
from utils import postprocess_qa_predictions, write_results_to_file
from constants import SQUAD, MLQA, SAVED_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=["bert", "xlm", "muril", "xlmr"], help="BERT or XLM")
    parser.add_argument('task', choices=["xlt", "gxlt"], help="XLT or G-XLT task")
    # parser.add_argument('train', choices=["squad", ""], help="SQuAD")
    parser.add_argument('lang_q', choices=["en", "hi", "es"], help="Language of the question - en, es, hi")
    parser.add_argument('lang_c', choices=["en", "hi", "es"], help="Language of the answer - en, es, hi")

    args = parser.parse_args()

    name = args.model
    is_xlt = True
    has_fast = False  # Existence of a fast tokenizer in Hugging Face
    model_name = ""
    model_type = ""
    lang_q = args.lang_q  # Question Language
    lang_c = args.lang_c  # Context Language

    match name:
        case "bert":
            model_type = "bert"
            model_name = "bert-base-multilingual-cased"
            has_fast = True
        case "xlm":
            model_type = "xlm"
            model_name = "xlm-mlm-tlm-xnli15-1024"
        case "muril":
            model_type = "bert"
            model_name = "google/muril-base-cased"
        case "xlmr":
            model_type = "roberta-base"
            model_name = "xlm-roberta-base"
            has_fast = True

    if args.task == 'gxlt':
        is_xlt = False

    if not is_xlt and lang_q == lang_c:
        sys.exit("G-XLT task requires question and answer to be in different languages")

    squad = load_dataset(SQUAD)
    mlqa = load_dataset(MLQA, "{}.{}.{}".format(MLQA, lang_q, lang_c))  # Eg. "mlqa.en.en"

    # TODO: Figure out a better way to save and load models
    # model_save_path = "{}_{}".format(name, SAVED_MODEL)
    # if os.path.exists(model_save_path):
    # model = AutoModelForQuestionAnswering.from_pretrained(model_save_path).to(device)
    # else:

    if has_fast:
        clqa = FastCLQA(model_name, model_type, squad, mlqa["test"])
        clqa.train_model()
        predictions = clqa.predict_model()
        final_predictions = postprocess_qa_predictions(mlqa["test"], clqa.tokenized_eval, predictions.predictions)
    else:
        clqa = NonFastCLQA(model_name, model_type, squad, mlqa["test"])
        clqa.train_model()
        final_predictions = clqa.predict_model()

    write_results_to_file(file_path="{}_{}_{}.json".format(lang_q, lang_c, name), text=final_predictions)


if __name__ == "__main__":
    main()
