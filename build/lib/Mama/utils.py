import random
import string
import math
import logging
import json
import os
import shutil

DB_URL = 'database/db.json'

def generate_random_token(len) -> str :
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(len))
    return result_str

def bytes_to_size_1024(bytes, precision=2):
    unit = ['B', 'KB', 'MB']
    i = int((bytes == 0) or (bytes and 1024 ** (1 + int(math.log(bytes, 1024)))))
    return f"{round(bytes / (1024 ** i), precision)} {unit[i]}"

# ------------------------------------------------------------
# VECTOR STOR HELPERS
# ------------------------------------------------------------
'''
def load_query_engine(srcdir, prompt) -> ConversationalRetrievalChain:
    db = ""
    if os.path.exists(srcdir) :
        embeddings = OpenAIEmbeddings()
        db = FAISS.load_local(srcdir, embeddings=embeddings)
    else:
        return None
    
    #use the faiss vector store we saved to search the local document
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

    prompt = PromptTemplate.from_template(prompt)
    qa = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(),
                                                retriever=retriever,
                                                condense_question_prompt=prompt,
                                                return_source_documents=True,
                                                verbose=False)
    return qa
'''
def get_kbs():
    db = get_db()
    if not db:
        return []
    return db.get("KBs", [])

def get_user(user_id) -> dict :
    db = get_db()
    if db:
        users = db.get("users", [])
        
        # Utilizza una list comprehension per trovare l'utente desiderato
        matching_users = [user for user in users if user.get("user_id") == user_id]
        
        if matching_users:
            return matching_users[0]

    logging.info(f"No user found with id: {user_id}")
    return {}

def get_session(user_id, session_id):
    user = get_user(user_id=user_id)
    if user: 
        sessions = user.get("sessions")
        if not sessions:
            return None
        
        for session in sessions:
            if session["id"] == session_id:
                return session
    return None

def save_session(db, user_id, session_id, in_session):
    users = db.get("users", [])
    
    for idx_user, user in enumerate(users):
        if user.get("user_id") == user_id:
            sessions = user.get("sessions", [])
            for idx_session, session in enumerate(sessions):
                if session["id"] == session_id:
                    db["users"][idx_user]["sessions"][idx_session] = in_session
                    break
    
    with open(DB_URL, 'w', encoding='utf-8') as file:
        json.dump(db, file, ensure_ascii=False, indent=4)

def modify_session(user_id, session_id, title=None) :
    if not title:
        return ""
    
    db = get_db()
    if not db:
        return ""
    
    users = db.get("users", [])
    for idx_user, user in enumerate(users):
        if user.get("user_id") == user_id:
            sessions = user.get("sessions", [])
            for idx_session, session in enumerate(sessions):
                if session["id"] == session_id:
                    db["users"][idx_user]["sessions"][idx_session]["title"] = title
                    with open(DB_URL, 'w', encoding='utf-8') as file:
                        json.dump(db, file, ensure_ascii=False, indent=4)
                    return "Session Title Changed"
    return ""
    

def get_db() :
    try:
        logging.debug(os.listdir())
        with open(DB_URL, 'r', encoding='utf-8') as f:
            db = json.load(f)
            return db
    except FileNotFoundError:
        logging.info(f"No {DB_URL} found")
    except json.JSONDecodeError:
        logging.info(f"Error decoding JSON from {DB_URL}")
    return None

def save_db(db):
    with open(DB_URL, "w") as f:
        json.dump(db, f, indent=4)


def save_chat_history(user_id, session_id, chat_history):
    db  = get_db()

    session = get_session(user_id, session_id)
    if session:
        session["chat_history"] = chat_history

    save_session(db, user_id, session_id, session)


def new_session(user_id, kb_id):
    
    #Se viene passata la kb_id allora vuol dire che la sessione viene associata ad un kb esistente altrimenti 
    # crea una session id di lunghezza 4 utilizzando generate_random_token(len)
    type = ""
    title = "New User Chat"
    if not kb_id:
        token = generate_random_token(4)  # TODO: config
        type = 'USER'
    else:
        token = kb_id
        type = 'GENERAL'
        title = kb_id

    # definisci questo json
    session = {
        "id": token,
        "title": title,
        "kb_id": token,
        "type": type,
        "chat_history": []
    }

    # recupera il json complessivo
    db = get_db()
    if not db:
        return {"error": "no db found"}

    # cerca l'utente nel database
    user_found = None
    for user in db["users"]:
        if user["user_id"] == user_id:
            user_found = user
            break

    # se l'utente non esiste, restituisci un errore
    if not user_found:
        return {"error": "user not found"}

    # aggiungi la nuova sessione all'utente esistente
    user_found["sessions"].append(session)

    # salva le modifiche nel database
    with open(DB_URL, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=4)

    return session

def delete_session(user_id, session_id, kb_dir) -> bool :
    if not user_id or not session_id:
        logging.info("error E004 - Invalid parameters")
        return False

    db = get_db()
    if db:
        for user in db["users"]:
            if user["user_id"] == user_id:
                sessions = user["sessions"]
                for idx, session in enumerate(sessions):
                    if session["id"] == session_id and session["type"] == "USER":
                        try:
                            kb = session["kb_id"]
                            del sessions[idx]
                                # salva le modifiche nel database
                            with open(DB_URL, 'w', encoding='utf-8') as f:
                                json.dump(db, f, ensure_ascii=False, indent=4)
                            shutil.rmtree(kb_dir+"/"+kb)
                            return True
                        except Exception as e:
                            logging.info(f"error removing session. Error={e}")
    else:
        logging.info(f"No db found")
    
    logging.info(f"error removing session. No user or session found {user_id}, {session_id}")
    return False    

def read_configuration(item):
    try:
        with open(DB_URL, 'r', encoding='utf-8') as f:
            db = json.load(f)
    except:
        logging.info(f"No {DB_URL} found")
        return
    
    if not db:
        return ""
    
    config = db.get("config")
    return config.get(item, [])

def save_kb(kb_id, title, description):
    KBs = get_kbs()
    for kb in KBs:
        if kb.get("kb_id", "") == kb_id:
            return
    kb = {
        "kb_id": kb_id,
        "type": "GENERAL",
        "Title": title,
        "Description": description
    }
    db = get_db()
    if not db:
        return
    
    db.get("KBs", []).append(kb)
    save_db(db)



