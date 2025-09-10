from langchain.prompts import ChatPromptTemplate

class Prompts():
    ANSWER_BASED_ON_CONTEXT = ChatPromptTemplate.from_template(
        """Answer the question based on the following context:
        Format your response as a valid markdown document
        Context: {context}

        Question: {question}
        """
    )

    CITES_SOURCES = ChatPromptTemplate.from_template(
        """You are an expert in answering questions with context from research papers.
        If your context is not relevant to question, say you do not have the valuable context.
        Answer the question based on the following context:
        Format your response as a valid markdown document
        Context: {context}

        Question: {question}
        """
    )
