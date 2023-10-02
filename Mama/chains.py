from langchain import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.chains import RetrievalQA
from utils import get_session, save_chat_history
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from Mama.cbLLM import cbLLM
import logging
import os
from flask import make_response

### https://python.langchain.com/docs/use_cases/question_answering/ 
def get_response(user_id, session_id, input_text, kb_dir, chat_history_len) :

    ##  --------------------------------------------------------------------------------------------------------------------------------
    ##1 Load Retriever. Retiever can be dynamically configured to access different sources. db.json containes a parameter: Retriever_type
    ##  --------------------------------------------------------------------------------------------------------------------------------
    chat_history = []

    session = get_session(user_id, session_id)
    if not session:
        return make_response("no session", 400)

    kb = kb_dir + "/" + session["kb_id"]
    if not kb:
        errMsg = f"ERR003. Non ho potuto caricare la Knowledge base {kb}"
        logging.info(errMsg)
        return  make_response(errMsg, 500)
    
    db = ""
    if os.path.exists(kb) :
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.load_local(kb, embeddings=embeddings)
    else:
        return None

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2}) #TODO: Use dynamic retriever from configurations
    #documents = retriever.get_relevant_documents(query=input_text)

    ##  --------------------------------------------------------------------------------------------------------------------------------
    ##2 Reconstruct Memory
    ##  --------------------------------------------------------------------------------------------------------------------------------
    chat_array = session["chat_history"]
    # Se ci sono più di N conversazioni, manteniamo solo le ultime 20
    if len(chat_array) > chat_history_len:
        chat_array = chat_array[- chat_history_len:]

    retrieved_messages = messages_from_dict(chat_array)
    memory = ConversationBufferMemory(chat_memory=ChatMessageHistory(messages=retrieved_messages), memory_key="history", input_key="question")
    memory.parse_obj(chat_array)

    ##  --------------------------------------------------------------------------------------------------------------------------------
    ##3 Create LLM
    ##  --------------------------------------------------------------------------------------------------------------------------------
    cb_llm = cbLLM()
    if not cb_llm:
        errMsg = f"ERR003. Non ho potuto caricare la LLM"
        logging.info(errMsg)
        ret = {
            "answer": errMsg,
            "documents" : [],
            "chat_history" : []
         }
        return ret
    
    llm = cb_llm.get_llm()
    if not llm:
        errMsg = f"ERR003. Non ho potuto caricare la LLM"
        logging.info(errMsg)
        ret = {
            "answer": errMsg,
            "documents" : [],
            "chat_history" : []
         }
        return ret
    
    ##  --------------------------------------------------------------------------------------------------------------------------------
    ##4 Create Prompt
      ##2.1 Il Prompt deve contenere {input_documents{page_content, source}}, {history}, {question} la risposta deve riportare i link 
      ##### così da fornire gli esatti link che ha usato la LLM. 
      ##### {history} è la memory_key di ConversationBufferMemory
    ##  --------------------------------------------------------------------------------------------------------------------------------
    prompt = "based on the context please return an answer to user question. I you don't know the answer simply say I don't know. ANSWER:"
    template = cb_llm.get_prompt_template()
    input_variables = []
    if template:
        input_variables = cb_llm.get_input_variables()
    
    prompt = PromptTemplate(template=template, input_variables=input_variables)
    logging.info(prompt)

    
    ##chain = load_qa_chain(llm, chain_type="stuff", memory=memory)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, 
        chain_type="stuff", 
        return_source_documents=True, 
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": True,
            "memory" : memory
        })
    response = qa_chain({"query": input_text})

    ##5 Save Memory
    memory.chat_memory.add_user_message(input_text)
    memory.chat_memory.add_ai_message(response["result"])
    dict = messages_to_dict(memory.chat_memory.messages)
    save_chat_history(user_id, session_id, dict)

    ##6 Return Result
    json_docs = []
    docs = response["source_documents"]
    for document in docs:
       json_docs.append({
           "page_content":document.page_content,
           "source" : document.metadata["source"]
        })
    ret = {
        "answer": response["result"],
        "documents" : json_docs,
        "chat_history" : []
    }
    return ret