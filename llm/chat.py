from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import GROQ_API_KEY

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)

prompt = ChatPromptTemplate.from_template("""
You are an Email Deliverability Expert.

Answer ONLY using the provided context.

Instructions:
- Do not repeat points.
- Merge similar information.
- Give a well-structured answer.
- Use bullet points.
- If the answer is not present in the context, say "I couldn't find this information."

Context:
{context}

Question:
{question}


Answer:
""")

chain = prompt | llm | StrOutputParser()

def generate_answer(question, context):
    return chain.invoke({
        "question": question,
        "context": context
    })