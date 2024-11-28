import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

load_dotenv()


def summarise(pdf_path):
    loader = PyPDFLoader(pdf_path)
    # Load and split the PDF into individual documents
    docs = loader.load_and_split()
    llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke(docs)
    return summary


if __name__ == "__main__":
    pdf_path = "neuroai.pdf"
    summary = summarise(pdf_path)
    print("\n\nSummary:")
    print(summary['output_text'])
