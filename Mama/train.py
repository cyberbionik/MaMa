import os 
from langchain import PromptTemplate
import shutil
import logging
from Mama.cbLLM import cbLLM
from Mama.utils import generate_random_token, get_session, save_kb
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from langchain.document_loaders.csv_loader import CSVLoader

def index_documents(folder) -> List[Document] :
    logging.info("Reading folder: "+folder)
    #loader = DirectoryLoader(folder, show_progress=True)
    loader = PyPDFDirectoryLoader(folder)
    docs = loader.load()
    logging.info(f"Read {len(docs)} documents")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    splits = splitter.split_documents(documents=docs)
    return splits

def train_on_documents(kb_dir, kb_id, src_dir="", documents = [], title="", description="", return_summary=False) -> str:
    summary = ""
    logging.debug(documents)

    if not documents or len(documents) == 0:
        logging.info("Reading folder: "+src_dir)
        loader = PyPDFDirectoryLoader(src_dir)
        docs = loader.load()
        logging.info(f"Read {len(docs)} documents")

        i = 0
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
        for document in docs:
            logging.info(f"Splitting doc {i+1}") 
            logging.debug(f"DOC={document}")
            s = splitter.split_documents(documents=[document])
            logging.info(f"generated {len(s)} splits for document {i+1}")
            logging.info(f"Saving doc nr. {i+1}")
            _save_vector_store(kb_dir, kb_id, s, title=title, description=description)
            i=i+1
    else:
        _save_vector_store(kb_dir, kb_id, documents, title=title, description=description)
    
    if return_summary == True and documents and len(documents) >0:
        try:
            llm = None
            try:
                llm = cbLLM().get_llm()
            except Exception as e:
                logging.info(f"Error loading llm {e}")
                return ""
            
            if not llm:
                logging.info("Cannot load LLM")
                return ""
            
            prompt_template = """Scrivi IN ITALIANO un sommario conciso e il link di partenza (SOURCE LINK) the seguente testo:
                "{text}"
                Inizia la risposta con "il presente documento"
                SOMMARIO CONCISO:
                SOURCE LINK:"""
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(llm, chain_type='map_reduce', map_prompt=prompt)
            summary = chain.run(documents[0])
            
            logging.info(f"---- SUMMARY PRODOTTO: {summary}")
            
        except Exception as e:
            logging.info(f"Errore nel caricare il summari del documento: {e}")

    return summary

def train_tmp_single_doc(user_id, sDir, kb_root_dir, session_id="", kb_id="") -> str :
    logging.debug(f"train_tmp_single_doc:: recevied params user_id={user_id}, session_id={session_id}, sDir={sDir}")

    sDocumentID = generate_random_token(16)
    sSrcDir = user_id+"-"+sDocumentID
    summary = ""
    
    logging.debug("mkdir: "+sSrcDir)
    try :
        os.mkdir(sSrcDir)
        logging.debug("copy document from: "+sDir+" to "+sSrcDir)
        shutil.move(sDir, sSrcDir)
    except Exception as e:
        logging.info(f"train_tmp_single_doc::exception during directory operations: {e}")
        return ""
    
    if session_id:
        session = get_session(user_id, session_id)
        if session:
            kb_id = session.get("kb_id", [])
        else:
            logging.debug("train_tmp_single_doc::invalid session")
            return ""

    if not kb_id:
        logging.debug("train_tmp_single_doc::invalid KB_ID")
        return ""
    
    summary = ""
    try:
        summary = train_single_doc(kb_root_dir, kb_id, sSrcDir)
        logging.debug("removing "+sDir)
        shutil.rmtree(sSrcDir)
    except Exception as e:
        logging.info(f"train_tmp_single_doc::exception during directory operations: {e}")
        
    return summary

def train_single_doc(kb_root_dir, kb_id, source_dir) -> str:
    summary = ""
    try:
        documents = index_documents(source_dir)
        if not documents:
            logging.info(f"Nessun documento trovato")
            return ""
        
        summary = train_on_documents(kb_root_dir, kb_id, src_dir="", documents=documents, title="", description="", return_summary=True)
        
    except Exception as e:
        logging.info(f"Errore nel caricare il documento: {e}")

    return summary

def train_on_csv(kb_root_dir, kb_id, title, description, filename, field_names, delimiter, quote_char, source_column):
    try:
        csv_args={
            'delimiter': delimiter,
            'quotechar': quote_char,
            'fieldnames': field_names
        }

        kb_path = kb_root_dir+"/"+kb_id

        loader = CSVLoader(file_path=filename, csv_args=csv_args, source_column=source_column)
        if not loader:
            logging.info(f"No Loader found")
            return False

        data = loader.load()

        if not data:
            logging.info(f"Nessun dato trovato")
            return False
        
        if os.path.exists(kb_path):
            faiss = FAISS.load_local(kb_path, embeddings=HuggingFaceEmbeddings())
            faiss.add_documents(documents=data)
           
        else:
            faiss = FAISS.from_documents(documents=data, embedding=HuggingFaceEmbeddings())

        logging.debug("Saving index...")
        if faiss:
            faiss.save_local(kb_path)
            save_kb(kb_id, title, description)
        else:
            logging.debug(f"Error saving on FAISS {kb_path}")
            return False

    except Exception as e:
        logging.info(f"Errore nel caricare il documento: {e}")
        return False

    return True


def _save_vector_store(kb_dir, kb_id, documents, title, description):
    embeddings =  HuggingFaceEmbeddings()
    kb_path = kb_dir+"/"+kb_id
    try:
        if os.path.exists(kb_path):
            faiss = FAISS.load_local(kb_path, embeddings=embeddings)
            faiss.add_documents(documents=documents)
           
        else:
            faiss = FAISS.from_documents(documents=[documents[0]], embedding=embeddings)

        logging.debug("Saving index...")
        faiss.save_local(kb_path)
        
        save_kb(kb_id, title, description)
     
    except Exception as e:
        logging.info(f"Errore nel salvare il document store: {e}")
