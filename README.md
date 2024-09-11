# experiments
Random experiments and prototypes, mostly AI models

## Structured Outputs 

It seems like there are multiple ways to get structured outputs: 

1. Run several iterations and use the one that fits the expected structure the best.
2. Use a model that supports a flag for JSON outputs. 
3. Use system prompt and other prompt engineering tricks like few shot examples to force the model to output the desired structure. From one blog post I read, it seems that flattening the structure works better than deep nested structures. For instance, if one wants output as "Ladies, Middle aged" for some image data, instead of having a separated Pydantic class for ladies and nesting an age structure inside that class, it is better to have a flat structure like "Ladies, Middle aged". 


`instructor` library is great as it is a thin wrapper around the OpenAI API and supports JSON outputs, but it does not seem to support two important things:

1. `transformers` models.
2. `regex`: For instance, if one wants the output to mimic something like "C|H+|O2", it does not seem possible to do this using Pydantic classes.

`transformer` seems to support tool calling, but only for models that support tool calling through a template:

[docs](https://huggingface.co/docs/transformers/main/chat_templating#advanced-tool-use--function-calling)

####  OpenRouter/Instructor

```python
client = instructor.from_openai(
       OpenAI(base_url="https://openrouter.ai/api/v1",
       api_key=os.environ["OPENROUTER_API_KEY"], ),
       mode=instructor.Mode.JSON,
)
```

[Source](https://github.com/jxnl/instructor/issues/676)

#### `transformers`

`outlines` library:

```python
import outlines

model = outlines.models.transformers("microsoft/Phi-3.5-vision-instruct")

prompt = "<s>result of 9 + 9 = 18</s><s>result of 1 + 2 = "
answer = outlines.generate.format(model, int)(prompt)
print(answer)
# 3

prompt = "sqrt(2)="
generator = outlines.generate.format(model, float)
answer = generator(prompt, max_tokens=10)
print(answer)
# 1.4142135

```

## Environment

```bash
conda create -n huggingface 
conda activate huggingface
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
conda install -c huggingface transformers
conda install -c conda-forge jupyterlab pandas numpy matplotlib scipy pip plotly scikit-learn seaborn streamlit gradio
conda install -c conda-forge spacy
pip install spacy-streamlit
pip install spacy-llm
pip install litellm openai tiktoken anthropic
pip install python-dotenv
```


## Training pipeline examples

Spacy has a few examples on how train more efficient models for particular tasks. For instance, to recognize fashion brands one can use the following example [project](https://github.com/explosion/projects/tree/v3/tutorials/ner_fashion_brands).
