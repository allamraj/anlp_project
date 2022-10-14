# Advanced NLP: Project
#
# Authors:
# A Aparajitha <aparajitha.allamraju@research.iiit.ac.in>
# Darshana S <darshana.s@research.iiit.ac.in>
#
# main.py

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=["bert", "xlm"], help="BERT or XLM")
    parser.add_argument('task', choices=["xlt", "gxlt"], help="XLT or G-XLT task")
    parser.add_argument('lang_q', choices=["en", "hi", "es"], help="Language of the question - en, es, hi")
    parser.add_argument('lang_a', choices=["en", "hi", "es"], help="Language of the answer - en, es, hi")

    args = parser.parse_args()

    is_bert = True
    is_xlt = True
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


if __name__ == "__main__":
    main()
