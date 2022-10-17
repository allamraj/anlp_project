# Advanced NLP: Project
#
# Authors:
# A Aparajitha <aparajitha.allamraju@research.iiit.ac.in>
# Darshana S <darshana.s@research.iiit.ac.in>
#
# models.py

from abc import ABC


class BaseModel(ABC):
    def __init__(self, lang1, lang2, model_name, model_type, train_set, val_set, eval_set):
        self.lang1 = lang1
        self.lang2 = lang2
        self.model_name = model_name
        self.model_type = model_type
        self.train_set = train_set
        self.val_set = val_set
        self.eval_set = eval_set
