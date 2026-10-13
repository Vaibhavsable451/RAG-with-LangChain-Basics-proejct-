import sys
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.config import get_llm, get_embeddings

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful assistant. Answer ONLY using the context below. "
     "If the context does not contain the answer, reply exactly: I don't know based on the provided documents. "
     "Cite chunk ids like [chunk 3]."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

def format_docs(docs):
    return "\n\n".join(f"[chunk {d.metadata.get('chunk_id')}] {d.page_content}" for d in docs)

def build_chain():
    vs = FAISS.load_local("index", get_embeddings(), allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | PROMPT | get_llm() | StrOutputParser()
    )
    return chain, retriever

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the remote work policy?"
    chain, retriever = build_chain()
    print("Q:", q)
    print("A:", chain.invoke(q))
    print("\nSources:")
    for d in retriever.invoke(q):
        print(f"  - chunk {d.metadata['chunk_id']}: {d.page_content[:80]!r}...")
