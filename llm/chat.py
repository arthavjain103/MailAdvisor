from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import GROQ_API_KEY
import os

# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ["GROQ_API_KEY"]
)
prompt = ChatPromptTemplate.from_template("""
You are an Email Deliverability Expert and Email Writing Assistant.

Answer ONLY using the provided context.

Rules:

1. If the user asks a question (e.g. "What is SPF?", "How do I improve email deliverability?"),
   provide a clear, accurate, well-structured answer using only the context.

2. If the user asks for an email, email template, draft, cold email, job application email,
   follow-up email, or any writing example:
   - Find the most relevant template from the context.
   - Return it as a COMPLETE email.
   - Do NOT summarize it.
   - Do NOT describe its sections.
   - Do NOT convert it into bullet points.
   - Preserve the original structure and formatting.
   - If the retrieved template is slightly incomplete, complete the missing part naturally while keeping the same tone and intent.
   - If multiple templates are retrieved, use only the best matching one.

3. Never mix multiple templates into one answer.

4. Never invent information that is not supported by the context, except for completing an obviously truncated template.

5. If the answer cannot be found in the context, reply exactly:
"I couldn't find this information."

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