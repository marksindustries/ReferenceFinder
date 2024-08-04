import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fpdf import FPDF
import arxiv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM


@st.cache_resource
def get_llm():
    return ChatGroq(model_name="mixtral-8x7b-32768")


llm = get_llm()

# Streamlit UI setup
st.title("Mango Reference Finder")
query = st.text_input("Keywords:")
references_count = st.number_input(
    "How Many:", min_value=1, max_value=50, step=1, value=5)

# Prompt templates
reference_template = """
You are an expert in Harvard style referencing.
Based on the provided title information {title}, create the corresponding Harvard style reference.
Ensure the reference is accurate and follows the Harvard style format precisely.
Only include the reference and NOTHING ELSE.
Output:
"""
reference_prompt = PromptTemplate.from_template(template=reference_template)

arrange_template = """
You will be provided a list of unordered Harvard references {references}.
You are required to arrange the list in ALPHABETICAL ORDER.
Output:
"""
arrange_prompt = PromptTemplate.from_template(template=arrange_template)

# Create LLMChains
reference_chain = LLMChain(llm=llm, prompt=reference_prompt)
arrange_chain = LLMChain(llm=llm, prompt=arrange_prompt)

# Function to fetch Arxiv papers


def fetch_arxiv_papers(query, max_results):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    return list(client.results(search))

# Process function to generate references


def process_references(papers):
    return [reference_chain.run(title=paper.title) for paper in papers]

# Function to arrange the list of references


def arrange_references(ref_list):
    return arrange_chain.run(references="\n".join(ref_list))

# Function to generate PDF


def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, "Mango Reference Finder", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('Arial', size=12)
    pdf.multi_cell(0, 10, text)
    pdf_path = "references.pdf"
    pdf.output(pdf_path)
    return pdf_path


# Streamlit button to trigger the fetching process
if st.button("Fetch"):
    with st.spinner("Fetching and processing references..."):
        try:
            # Fetch papers
            papers = fetch_arxiv_papers(query, references_count)

            if not papers:
                st.warning("No papers found for the given query.")
            else:
                # Process references
                references = process_references(papers)

                # Arrange references
                ordered_references = arrange_references(references)

                # Display the references
                st.subheader("Ordered References:")
                st.text(ordered_references)

                # Generate and provide PDF download
                pdf_path = generate_pdf(ordered_references)
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="Download PDF",
                        data=file,
                        file_name="references.pdf",
                        mime="application/pdf"
                    )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add some information about the app
st.sidebar.title("About")
st.sidebar.info(
    "This app fetches academic papers from Arxiv based on your keywords, "
    "generates Harvard style references, and arranges them alphabetically. "
    "You can then download the references as a PDF."
)
