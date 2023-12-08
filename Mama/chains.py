import logging
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.chains import RetrievalQA
from Mama.utils import get_session, save_chat_history
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from Mama.cbLLM import cbLLM
from Mama.config import Configuration

### https://python.langchain.com/docs/use_cases/question_answering/ 

def get_response(user_id, session_id, input_text, kb_dir, chat_history_len) :
    """
    Gets an input text and generate a response from GenAI.

    STEP 1. Uses a FAISS vector store to retrieve documents
    STEP 2. Reconstruct Memory
    STEP 3. Dynamically creates the LLM (from configuration file db.json)
    STEP 4. Uses RetrievalQA to generate the response

    Parameters:
        user_id: user_id associated to the conversation
        session_id: session_id is the Knowledge Base name associated to the session (present in config file db.json)
        kb_dir: directory where the Knoweldge Base is located
        chat_history_len : max lenght for memory. After that chat_history is cutted off starting from the older one

    Returns:
        {
            "answer": the answer in string format,
            "documents" : a JSON with this syntax: {"page_content":extract from retrieved document, "source" : the source of the document}
            "chat_history" : [] (TODO:)
        }

    configurations (from db.json):
        search_type: serach_type parameter for retriever
        num_docs: number of docs to retrieve from Vector Store
        prompt: from db.json
    """

    ##  --------------------------------------------------------------------------------------------------------------------------------
    ##1 Load Retriever. Retiever can be dynamically configured to access different sources. db.json containes a parameter: Retriever_type
    ##  --------------------------------------------------------------------------------------------------------------------------------
    chat_history = []

    session = get_session(user_id, session_id)
    if not session:
        return _err_msg("No Session")

    kb = kb_dir + "/" + session["kb_id"]
    if not kb:
        return _err_msg(errMsg = f"ERR003. Error Loading Knowledge base {kb}")
    
    db = ""
    if os.path.exists(kb) :
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.load_local(kb, embeddings=embeddings)
    else:
        return _err_msg(errMsg = f"ERR003. Error Loading Knowledge base {kb}")

    config = Configuration()
    search_type = config.get("search_type")
    if not search_type:
        search_type = "similarity"
    num_docs = config.get("num_docs")
    if not num_docs:
        num_docs = 2
    retriever = db.as_retriever(search_type=search_type, search_kwargs={"k":num_docs})
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
        return _err_msg( errMsg = f"ERR003. Error Loading LLM")
        
    llm = cb_llm.get_llm()
    if not llm:
        return _err_msg( errMsg = f"ERR003. Error loading LLM")
    
    ##  --------------------------------------------------------------------------------------------------------------------------------
    ##4 Create Prompt
      ##2.1 Il Prompt deve contenere {input_documents{page_content, source}}, {history}, {question} la risposta deve riportare i link 
      ##### così da fornire gli esatti link che ha usato la LLM. 
      ##### {history} è la memory_key di ConversationBufferMemory
    ##  --------------------------------------------------------------------------------------------------------------------------------
    
    prompt = cb_llm.get_prompt_template()
    #input_variables = []
    #if template:
    #    input_variables = cb_llm.get_input_variables()
    
    #prompt = PromptTemplate(template=template, input_variables=input_variables)
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

def _err_msg( errMsg : str ) :
    logging.info(errMsg)
    ret = {
        "answer": errMsg,
        "documents" : [],
        "chat_history" : []
    }
    return ret