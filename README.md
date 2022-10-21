## Advanced NLP - Project
MLQA: Implementation of Cross Lingual Question Answering

### MLQA
MultiLingual Question Answering (MLQA) is a benchmark dataset released 
by Facebook for evaluating extractive cross lingual question answering.
 MLQA contains data in 7 languages, for the project the evaluation is 
being done in 3 viz. English, Spanish and Hindi

#### Usage
```bash
python ./main.py <model> <task> <lang_q> <lang_c>
```

```text
model = "bert", "xlm", "muril", "xlmr"
task = "xlt", "gxlt"
lang_q = "en", "hi", "es" (Question Language)
lang_c = "en", "hi", "es" (Context Language)
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

```

### Links to files
Saved Model - 
<a href="">Here</a>

Saved Model - 
<a href="">Here</a>