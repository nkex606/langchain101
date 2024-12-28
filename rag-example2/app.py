from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import List, TypedDict


class ResponseFormatter(BaseModel):
    """Use this tool to structure your response."""

    env: str = Field(description="env")
    addr: str = Field(description="ip address of the server")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: dict


def retrieve(state: State):
    retrieve_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieve_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = model.invoke(messages)
    return {"answer": response.model_dump()}


model = ChatOllama(model="llama3.2").with_structured_output(ResponseFormatter)

loader = JSONLoader(
    file_path="./env.json",
    jq_schema=".[]",
    text_content=False,
)
docs = loader.load()

embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)

prompt_template = """
    you are a helpful assistant that answers questions based on the context provided.
    If you can't find the answer in the context, just say "I can't find the answer in the context provided".
    Don't make up an answer.
    context: {context}
    question: {question}
    answer:
    """

prompt = ChatPromptTemplate.from_template(prompt_template)
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "What is ip address of a-server server in dev?"})
print(f'Context: {result["context"]}\n')
print("-------------------------------")
print(f'Answer: {result["answer"]}')
