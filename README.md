## Advanced NLP - Project
MLQA: Implementation of Cross Lingual Question Answering

### MLQA
MultiLingual Question Answering (MLQA) is a benchmark dataset released 
by Facebook for evaluating extractive cross lingual question answering.
 MLQA contains data in 7 languages, for the project the evaluation is 
being done in 3 viz. English, Spanish and Hindi

#### Usage
```bash
python ./main.py <model>
```

### Requirements
```bash
gensim
nltk
numpy
pandas
Pillow
six
torch
```
Requirements can bbe installed with
```bash
pip install -r requirements.txt
```

### Files in codebase
```bash
language_model.py - Main file
models.py - Language Models
data.py - Corpus Usage and Dataset
constants.py - Constants
utils.py - Utility Functions
```

### Links to files
Reduced glove embeddings - 
<a href="https://iiitaphyd-my.sharepoint.com/:u:/g/personal/aparajitha_allamraju_research_iiit_ac_in/EalVeVceFCJJnAERVEsa_R8Bt8dbZaiG2nu8eh16htpUlQ?e=WUbHYz">Here</a>

Saved Language Model 1 - 
<a href="https://iiitaphyd-my.sharepoint.com/:u:/g/personal/aparajitha_allamraju_research_iiit_ac_in/Ec710HxU1gtOgg-TbzZANFwBgRivdmOsQEsMBeQ0HlhQzQ?e=1KPJmP">Here</a>