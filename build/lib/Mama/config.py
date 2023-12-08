import logging
from typing import Any, Dict
from Mama.utils import get_db, save_db


class Configuration() :

    def __init__(self, initial_config=None):
        self.config = None
        self.prompt_template = ""
        self.input_variables = []
        try:
            self.load()
        except Exception as e:
            print(f"Error Loading Configuration {e}")
    
    def load(self) :
        logging.basicConfig(format='[%(asctime)s]::[%(levelname)s - %(funcName)s]:: %(message)s', filename='flask.log', level=logging.DEBUG)

        db = get_db()
        if not db:
            return False
        
        self.config = db.get("config")
        if not self.config:
            logging.info(f"No configuration session found in db.json")
            return False
        
        logging.info(f"Configuration: {self.config}")
        return True
    
    def get(self, name : str) -> Any:
        if not self.config:
            return ""
        return self.config.get(name, [])
    
    def set(self, name : str, data : Any):
        if not self.config:
            return
        self.config[name] = data
        self.save()
    
    def save(self):
        db = get_db()
        if not db:
            return False
        
        db["config"] = self.config
        save_db(db)

    def get_llm_params(self, model) -> Dict[str, Any]:      
        db = get_db()
        if not db:
            return {}
        
        LLMs = db.get("LLMs", [])
        if not LLMs:
            return {}
        
        for llm in LLMs:
            if model == llm.get("model", ""):
                self.prompt_template = llm.get("prompt_template", "")
                self.input_variables = llm.get("input_variables", "")
                return llm.get("parameters", {})
        return {}
    
    def add_chatbot(self, chatbot : Dict) :
        db = get_db()
        if not db:
            return {}
        
        cbs = db.get("CHATBOT", [])
        if not cbs:
            cbs = [chatbot]
        else:
            found = False
            for i, cb in enumerate(cbs):
                if cb['chat_id'] == chatbot['chat_id']:
                    found = True
                    cbs[i] = chatbot
                    break
            if not found:
                cbs.append(chatbot)
                            
        db["CHATBOT"] = cbs

        save_db(db)

    def get_chatbot(self, id)  -> [] :
        '''
        Returns Chatbot parameters from db.json
        if title == "" than returns all Chatbot
        '''
        db = get_db()
        if not db:
            return []
        
        cbs = db.get("CHATBOT", [])
        if not cbs:
            return []
        
        if not id:
            return cbs
        else:
            for chatbot in cbs:
                chat_id = chatbot.get("chat_id", "")
                if chat_id and id == chat_id:
                    return [chatbot]
        return []
        
    def get_prompt_template(self):
        return self.prompt_template
    def get_input_variables(self):
        return self.input_variables
    
