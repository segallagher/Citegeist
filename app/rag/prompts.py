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
        """Based only on the context provided, write a confident and well-supported answer to the question, as if summarizing insights for a technical literature review.

            You may draw reasonable inferences as long as they are clearly supported by the text. Avoid adding any external information not grounded in the context.

            Structure the answer like a mini literature review, including:
            - A brief overview of the problem
            - Relevant techniques or findings
            - Strengths, limitations, or comparisons
            - A concluding summary

            You may draw inferences or conclusions as long as they are supported by the text. Be precise and assertive. Avoid vague or overly cautious language.

            Do not include any information not supported by the context.
            ---

            Example

            Question: What are common methods for sentiment analysis in NLP?

            Context: The papers discuss multiple sentiment analysis techniques, including lexicon-based approaches like VADER and machine learning models such as logistic regression and BERT. VADER is noted for being lightweight and effective on social media text, while transformer-based models like BERT show higher accuracy on longer, more complex inputs.

            Answer: Sentiment analysis methods in NLP generally fall into two categories: lexicon-based and machine learning-based approaches. VADER is a widely used lexicon-based method known for its performance on short, informal text like tweets. In contrast, transformer models such as BERT offer superior accuracy on more complex language, though they require more computational resources. Overall, the choice of method depends on the task context, with BERT being preferred for high-accuracy applications.

            ---

            Question: {question}

            Context: {context}

            Answer:
        """
    )

    ANSWER_QUESTION_NO_CONTEXT = ChatPromptTemplate.from_template(
        """You are an expert in research topics. Answer the following question as accurately and clearly as possible, based on your existing knowledge.

        - Do not guess if you're unsure.
        - If relevant research is not known to you, say so honestly.
        - Do not hallucinate sources or details.

        Format your response in valid markdown.

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
        CONTEXT: {context}

        QUESTION: {question}

        RESPONSE: {response}

        Determine if the RESPONSE correctly and fully answer the QUESTION, using only the provided CONTEXT or established knowledge

        Your grading criteria for a good RESPONSE are as follows:
        - Is consistent with the retrieved academic content
        - Answers all key parts of the question
        - Does not invent facts or citations
        - Acknowledges uncertainty where appropriate
        Do NOT penalize a response for not answering if the information was not available in the context.

        Your answer must begin with either "Yes |" or "No |" or "N/A" followed by your reasoning.

        Examples:
        Yes | The response answers the question fully and accurately using the context
        No | The response does not answer the question, and the answer is present in the context
        N/A | The response correctly states that the answer is not found in the context
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

    MULTI_QUERY = ChatPromptTemplate.from_template(
        """
        You are an AI language model assistant. Your task is to generate {n_prompts}
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help the user
        overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by only one newline character. Only provide the requested
        perspectives without any additional text. Original question: {question}
        """
    )
