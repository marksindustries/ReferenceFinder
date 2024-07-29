import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.retrievers import ArxivRetriever
from langchain_community.document_loaders import ArxivLoader
from langchain.prompts import PromptTemplate
from fpdf import FPDF
# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(
    model_name="mixtral-8x7b-32768"
)

# Streamlit UI setup
st.title("Reference Finder")

query = st.text_input("Keywords:")
references_count = st.number_input("How Many:", min_value=1, step=1)

# Prompt template for referencing
template = """
You are an expert in Harvard style referencing. 
Based on the provided title information {title}, create the corresponding Harvard style reference. 
Ensure the reference is accurate and follows the Harvard style format precisely.
Only include the reference and NOTHING ELSE.

Output:
"""
prompt = PromptTemplate.from_template(template=template)

# Prompt template for arranging references in alphabetical order
arrange_template = """
You will be provided a list of unordered Harvard references {list}. 
You are required to arrange the list in ALPHABETICAL ORDER.

Output:
"""
arrange_prompt = PromptTemplate.from_template(template=arrange_template)

# Arxiv loader setup
loader = ArxivLoader(
    query=query,
    load_max_docs=references_count
)

# Process function to invoke chain and gather references


def process(docs):
    answers = []
    for doc in docs:
        response = chain.invoke({"title": doc.metadata})
        # Adjust this line based on actual response structure
        answers.append(response.content)
    return answers

# Function to arrange the list of references


def arrange(ref_list):
    response = arrange_chain.invoke({"list": ref_list})
    return response.content


# Function to generate PDF
def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(10, 10)

    # Add a title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt="Harvard Style References", ln=True, align='C')
    pdf.ln(10)  # Add a line break

    # Reset font for the main content
    pdf.set_font('Arial', size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, txt=line)

    pdf_path = "./references.pdf"
    pdf.output(pdf_path)
    return pdf_path


# Streamlit button to trigger the fetching process
if st.button("Fetch"):
    docs = loader.load()
    chain = prompt | llm
    arrange_chain = arrange_prompt | llm

    answers = process(docs)
    ordered_references = arrange(answers)

    # Display the references
    st.write(ordered_references)

    # Provide option to download the PDF
    pdf_path = generate_pdf(ordered_references)
    with open(pdf_path, "rb") as file:
        btn = st.download_button(
            label="Download PDF",
            data=file,
            file_name="references.pdf",
            mime="application/pdf"
        )
