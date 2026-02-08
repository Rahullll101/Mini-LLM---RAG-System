# Rag_app.py
from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        ##Persona and objective
        "You are an assistant who understands context and answers to questions"
        "**If the answer is not found, say: Information not found in the provided documents.***"
        ##critical operation rule
        "- never mention you are ai "
        "- try to be more accurate with provided context"
        "- answer only if context is provided to you"
        "- never answer to query like :- Based on the context provided ......"
        "- never answer out of the scope/context (example:- query= where is belagavi , answer ='Information not found in the provided documents ' "
    
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])


def answer_query(query, vector_db, llm, k=3, metadata_filter=None):
    if metadata_filter:
        docs = vector_db.similarity_search(
            query, k=k, filter=metadata_filter
        )
    else:
        docs = vector_db.similarity_search(query, k=k)

    context = "\n\n".join(doc.page_content for doc in docs)

    messages = RAG_PROMPT.format_messages(
        context=context,
        question=query
    )

    answer = llm.invoke(messages)
    return answer, docs
