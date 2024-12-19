# build an agent

follow this [reference](https://python.langchain.com/docs/tutorials/agents/) to build an agent for your LLM

## prepare api keys

before running this app, you should have these api keys prepared:

  - LANGCHAIN_API_KEY (if you want to trace on LangSmith)
  - TAVILY_API_KEY (change this to your prefered search engine)
  - OPENAI_API_KEY (change this to your prefered model)

## how to run

install the dependencies

```bash
$ pip3 install -r requirements.txt
```

run the app

```bash
$ python3 app.py
```