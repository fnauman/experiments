# Langchain: Crash course


## Definitions

- **LLM Wrappers**: Libraries that offer a nice unified interface to a variety of LLMs, both open and closed source. 
- **Indexer**: A program that indexes a corpus of text, and allows for fast lookup of words and phrases. Mostly used for `retrieval`.
- **Chains** (or pipelines): Predefined sequence of tasks for LLMs. For example, a chain could be `tokenize -> lemmatize -> pos-tag -> ner -> parse -> sentiment -> ...`.
- **Agents**: In contrast to a `chain`, an agent makes the decisions and designs the chain. 


## Problems faced so far

- **Open source** LLMs are hard to call using LangChain. In particular, **open source** LLMs that can be run on a laptop/desktop are hard to find instructions for in the documentation.
- **Closed source** Most tutorials almost exclusively focus on `OpenAI`'s `GPT-3.5` and `GPT-4`, which require an API key and are costly. The only positive is that one doesn't need a powerful GPU to run this version.


## Sources

- `freeCodeCamp`: [Tutorial](https://www.youtube.com/watch?v=lG7Uxts9SXs&ab_channel=freeCodeCamp.org)
