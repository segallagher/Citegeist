from langchain.prompts import ChatPromptTemplate

class Prompts():
    ANSWER_BASED_ON_CONTEXT = ChatPromptTemplate.from_template(
        """Answer the question based on the following context:
        {context}

        Question: {question}
        """
    )
