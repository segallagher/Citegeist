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
        """Answer the following question by synthesizing insights from recent research papers. Structure the answer like a mini literature review, with:

        - A brief overview of the problem
        - A comparative summary of proposed techniques or evaluations
        - Key findings or contributions from papers (with citations)
        - Strengths and limitations noted by the papers
        - A concluding summary

        Use only the context provided below.
        Question: {question}

        Context: {context}

        Answer:
        """
    )

    ANSWER_QUESTION_NO_CONTEXT = ChatPromptTemplate.from_template(
        """You are an expert in answering questions with about research papers.
        If do not know the answer to the question, say you do not know and do not guess at the answer.
        Format your response as a valid markdown document
        Answer the question:

        Question: {question}
        """
    )

    IS_DOCUMENT_RELEVANT = ChatPromptTemplate.from_template(
        """
        Evaluate the relevance of the CONTEXT in answering the QUESTION.
        A relevant text contains information that helps answer the question, even if partially.
        Return one of the following labels: 'Relevant', or 'Irrelevant'
        Do not return any other text besides 'Relevant', or 'Irrelevant'
        QUESTION: {question}

        CONTEXT: {context} 
        """
    )

    DOES_RESPONSE_ANSWER_QUESTION = ChatPromptTemplate.from_template(
        """
        Determine if the following RESPONSE answers the QUESTION sufficiently.
        A response that answers the question will answer each part of the question fully.
        Your answer will start with Yes or No followed by the symbol |
        After the | symbol you will provide your reasoning for your Yes or No answer
        Always start your response with either "Yes |" or "No |" followed by your reasoning
        QUESTION: {question}

        RESPONSE: {response}
        """
    )

    CHOOSE_BETTER_RESPONSE = ChatPromptTemplate.from_template(
        """
        You are reviewing two answers (labeled 'answer 1' and 'answer 2') to a research-oriented question. Your goal is to identify which answer better synthesizes academic research to inform the question.

        Evaluate based on:
        - Depth of research insight
        - Relevance and accuracy of cited work
        - Clarity of synthesis (not just listing papers)
        - Critical analysis and comparison
        - Use of evidence from the context
        Indicate which answer you prefer by starting your response with either 'answer 1 |' or 'answer 2 |' followed by an explanation for your choice

        QUESTION: {question}

        Answer 1: {answer1}

        Answer 2: {answer2}
        """
    )
