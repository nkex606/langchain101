# rag

follow [Build a Retrieval Augmented Generation (RAG) App: Part 1](https://python.langchain.com/docs/tutorials/rag/) to build a RAG application

This app try to find server ip address recording to `env.json` file.
The origin reponse from model is a `AIMessage`, but to simplify the response, I use a `ResponseFormatter` to convert it to a `dict` type.

## install ollama on your localhost

This app use `llama3.2` as a default model, you may have to install ollama first.
You can also change to other model you preffered.

## how to run

install the dependencies

```bash
$ pip3 install -r requirements.txt
```

run the app

```bash
$ python3 app.py
```