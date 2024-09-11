# Spacy: NER using pretrained models, training new models and using `spacy-llm`

## Installation

```bash
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
pip install spacy-llm
pip install spacy-streamlit
```


## TODO

The key task is **Named Entity Recognition**: Once an audio file is processed to generate the transcript, different attributes of the cloth need to be read in. For example, the following sentence: 

> "A blue shirt with a collar and long sleeves, condition 3, pilling 1, size xl, 100% cotton, Lager 157, made in Bangladesh"

should be parsed into the following attributes:

- `color`: blue
- `type`: shirt
- `condition`: 3
- `pilling`: 1
- `size`: xl
- `material`: 100% cotton
- `brand`: Lager 157

Spacy `matcher` can be used to match the attributes. A proof of concept is shown in [ner_intro.ipynb](spacy_ner/ner_intro.ipynb).


## Other Resources

[GLiNER - Base](https://huggingface.co/spaces/tomaarsen/gliner_base).
