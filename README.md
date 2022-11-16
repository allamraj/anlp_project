## Advanced NLP - Project
MLQA: Implementation of Cross Lingual Question Answering

### MLQA
MultiLingual Question Answering (MLQA) is a benchmark dataset released 
by Facebook for evaluating extractive cross lingual question answering.
 MLQA contains data in 7 languages, for the project the evaluation is 
being done in 3 viz. English, Spanish and Hindi

#### Usage
Running the models
```bash
python ./main.py <model> <task> <lang_q> <lang_c> <top>
```

```text
model = "bert", "xlm", "muril", "xlmr"
task = "xlt", "gxlt"
lang_q = "en", "hi", "es" (Question Language)
lang_c = "en", "hi", "es" (Context Language)
top = "1", "3", "20" (Top k best answers)
```
Running the MLQA evaluation
v1 is the original evaluation for MLQA. v2 does the evaluation for top k best answers
```bash
python mlqa_evaluation_v<1/2>.py \
   path/to/MLQA_V1/test-context-<lang_c>-question-<lang_q>.json \
   path/to/predictions.json \
   <lang_c>
```

### Requirements
```bash
torch
transformers
datasets
sacremoses
simpletransformers
```
Requirements can bbe installed with
```bash
pip install -r requirements.txt
```

### Files in codebase
```text
main.py - Main file
models.py - Base Model
clqa_fast.py - Fast Tokenizer based models like BERT and XLMR
clqa_nonfast.py - Python Tokenizer based models like XLM and MuRIL
constants.py - constants
utils.py - Utility Functions
mlqa_evaluation_v1.py - Facebook's evaluation script for MLQA
mlqa_evaluation_v2.py - Evaluation script fot top k answers

```

### Links to files
Fine-tuned BERT - 
<a href="https://huggingface.co/darshana1406/bert-base-multilingual-cased-finetuned-squad">Here</a>

Fine-tuned XLM - 
<a href="https://iiitaphyd-my.sharepoint.com/:u:/g/personal/aparajitha_allamraju_research_iiit_ac_in/ERZStpKmgwNAh1V3t8JekPABMXpL75CE2KbxpJNzpriKgQ?e=jq5b5e">Here</a>

Fine-tuned MuRIL - 
<a href="https://iiitaphyd-my.sharepoint.com/:u:/g/personal/aparajitha_allamraju_research_iiit_ac_in/EYXUdbaskapHiMNB6J4JavsB93NjxmpBIp7D5ujePoMO0w?e=lWr2eP">Here</a>

Fine-tuned XLM-R - 
<a href="https://huggingface.co/darshana1406/xlm-roberta-base-finetuned-squad">Here</a>


### Reference
P Lewis, B Oğuz, R. Rinot, S. Riedel and H. Schwenk MLQA: Evaluating Cross-lingual Extractive Question Answering

