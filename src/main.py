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
from clqa_xlm import CLQAXLM
from utils import write_results_to_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=["bert", "xlm"], help="BERT or XLM")
    parser.add_argument('task', choices=["xlt", "gxlt"], help="XLT or G-XLT task")
    # parser.add_argument('train', choices=["squad", ""], help="SQuAD")
    parser.add_argument('lang_q', choices=["en", "hi", "es"], help="Language of the question - en, es, hi")
    parser.add_argument('lang_a', choices=["en", "hi", "es"], help="Language of the answer - en, es, hi")

    args = parser.parse_args()

    is_bert = True
    is_xlt = True
    train_data = 'squad'
    eval_data = 'mlqa'
    lang_q = args.lang_q
    lang_a = args.lang_a

    if args.model == 'xlm':
        is_bert = False
    if args.task == 'gxlt':
        is_xlt = False

    if not is_xlt and lang_q == lang_a:
        sys.exit("G-XLT task requires question and answer to be in different languages")

    model_type = "bert" if is_bert else "xlm"
    model_name = "bert-base-multilingual-cased" if is_bert else "xlm-mlm-tlm-xnli15-1024"
    train = load_dataset(train_data)
    evalset = load_dataset(eval_data, "{}.{}.{}".format(eval_data, lang_q, lang_a))  # Eg. "mlqa.en.en"

    if is_xlt:
        clqa = CLQAXLM(lang_q, lang_a, model_name, model_type, train, evalset, evalset)
        clqa.config_model()
        clqa.train_model()
        answers, probs = clqa.predict_model()
        write_results_to_file('{}_{}.json'.format(lang_q, lang_a), answers)


if __name__ == "__main__":
    main()
