# =========================================================
# QA PDF RAG Chatbot
# =========================================================

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# =========================================================
# Global LLM (load once)
# =========================================================
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

hf_pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.0
)

LLM = HuggingFacePipeline(pipeline=hf_pipe)

# =========================================================
# Build RAG Pipeline
# =========================================================
def build_qa_from_pdf(file):
    if file is None:
        return None, "❌ No PDF uploaded."

    try:
        # 1. Load PDF
        loader = PyPDFLoader(file.name)
        documents = loader.load()

        # 2. Split text (PDF-friendly)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # 3. Embeddings (CPU safe)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # 4. Vector DB (no persistence — Spaces safe)
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # 5. Retriever (MMR, small k)
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "lambda_mult": 0.7}
        )

        # 6. Prompt (strict grounding)
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Answer the question using ONLY the context below.
If the answer cannot be found in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
        )

        # 7. QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        return qa_chain, "✅ PDF processed successfully! Ask questions below."

    except Exception as e:
        return None, f"❌ Error processing PDF: {str(e)}"

# =========================================================
# Ask Question
# =========================================================
def ask_question(question, qa_chain):
    if qa_chain is None:
        return "⚠️ Upload and process a PDF first."

    if not question or len(question.strip()) < 5:
        return "Ask a more specific question."

    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]

        sources = sorted({
            f"{doc.metadata.get('source', 'Unknown')} "
            f"(Page {doc.metadata.get('page', 0) + 1})"
            for doc in result["source_documents"]
        })

        sources_text = "\n".join(sources)
        return f"{answer}\n\n📄 Sources:\n{sources_text}"

    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

# =========================================================
# Gradio UI
# =========================================================
with gr.Blocks(title="📄 PDF RAG Chatbot") as demo:
    gr.Markdown("## 📄 Chat with your PDF (RAG)")

    qa_state = gr.State(None)

    pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
    process_btn = gr.Button("Process PDF")
    status = gr.Textbox(label="Status", interactive=False)

    question = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="Answer", lines=8)

    process_btn.click(
        build_qa_from_pdf,
        inputs=pdf_file,
        outputs=[qa_state, status]
    )

    question.submit(
        ask_question,
        inputs=[question, qa_state],
        outputs=answer
    )

    gr.Button("Ask").click(
        ask_question,
        inputs=[question, qa_state],
        outputs=answer
    )

demo.launch()
