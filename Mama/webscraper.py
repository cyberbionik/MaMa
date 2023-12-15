import json
import logging
import os.path
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag
from langchain.document_loaders import AsyncChromiumLoader # type: ignore
from langchain.document_transformers import BeautifulSoupTransformer # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib3 import disable_warnings
from Mama.train import train_on_documents
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from urllib3.exceptions import InsecureRequestWarning

def links_crawler(root_url, base_url, depth, tag_name, attributes):
    if not root_url or not depth:
        logging.info("No root_url")
        return

    logging.info("------------------------------------------------------------")
    logging.info(f"STARTING CRAWLING LINKS FOR {root_url}")

    valid_links, discarded_links, ret_visited = _crawl(root_url=root_url, base_url=base_url, url=root_url, current_depth=depth, tag_name=tag_name, attributes=attributes)
   
    # Salva i link in file
    with open('crawled_links.json', 'w') as json_file:
        json.dump(list(valid_links), json_file)

    with open('invalid_links.json', 'w') as json_file:
        json.dump(list(discarded_links), json_file)
    
    size = len(valid_links)
    logging.info(f"total scraped links ={size}")
    logging.info("------------------------------------------------------------")

    return size

def process_url(link) :
    try:
        html = []
        try:
            loader = AsyncChromiumLoader([link])
            html = loader.load()
        except Exception as e:
            logging.info(f"error loading page {link} {e}")
            return None
        
        if len(html) > 0:
            #page = html[0].page_content
            # Transform
            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = bs_transformer.transform_documents(html)

            # Grab the first 1000 tokens of the site
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
            splits = splitter.split_documents(documents=docs_transformed)
            
            if len(splits) > 0:
               return splits
            else: 
                logging.debug("document skipped..." )
                return None
        else:
            logging.debug("document skipped..." )
            return None
    except Exception as e:
            logging.debug(f"General Error. Skipping... {e}" )
            return None

def process_single_url(kb_dir, kb_id, link):
    splits = []
    try:
        splits = process_url(link)
    except Exception as e:
        logging.info(f"error processing URL: {e}")
        return e
    
    summary = ""
    try:
        if splits and len(splits) >0 :
            summary = train_on_documents(kb_dir=kb_dir, kb_id=kb_id, documents=splits, title="", description="", return_summary=True)
        else:
            logging.info(f"error processing URL. No splits")
            return ""
    except Exception as e:
        logging.info(f"Errore nel caricare il documento: {e}")

    return summary

def process_urls(kb_dir, kb_id, title, description):
    skipped_links =  []
    splitted_links = []

    with open('crawled_links.json', 'r') as json_file:
        valid_links = json.load(json_file)

    for i, link in enumerate(valid_links, start = 1):
        try:
            logging.info("loading link ["+str(i)+"]:" + link)
            splits = process_url(link=link)
            if(splits):
                 splitted_links = splitted_links + splits
            else:
                 skipped_links.append(link)
        except:
            logging.debug("General Error skipping..." )
            skipped_links.append(link)
        
        logging.info(f"skipped following links: {skipped_links}")

    try:
        if len(splitted_links) >0 :
            train_on_documents(kb_dir=kb_dir, kb_id=kb_id, src_dir='', documents=splitted_links, title=title, description=description, return_summary=False)
        
    except Exception as e:
        logging.info(f"Errore nel caricare il le URLs: {e}")

    
def save_vector_store(documents, faiss_src:str) :
    try:
        if os.path.isdir(faiss_src):
            faiss = FAISS.load_local(faiss_src, embeddings=HuggingFaceEmbeddings())
            if faiss:
                faiss.add_documents(documents=documents)
            else:
                logging.info(f"error loading from: {faiss_src}")
        else:
            embeddings = HuggingFaceEmbeddings()
            #embeddings = OpenAIEmbeddings() #TODO: read from configurations
            faiss = FAISS.from_documents(documents=documents, embedding=embeddings)
    except Exception as e:
        logging.info(f"error adding document: {e}")
        return

    try:
        if not faiss:
            faiss.save_local(faiss_src)
            logging.debug("index saved...")
        else:
            logging.info("no index to save...")
    except Exception as e:
        logging.info(f"error saving index: {e}")

def isValidLink(link, root_url):
    """
    Verifica se un link è valido per il crawling.

    Parameters:
    - link: Il link da verificare.
    - root_url: L'URL root da usare come riferimento.
    - valid_links: Lista dei link già considerati validi.
    - discarded_links: Lista dei link scartati.

    Returns:
    - True se il link è valido per il crawling, False altrimenti.
    """
    # Controlla se il link è esterno
    if not link.startswith(root_url):
        return False

    return True

def build_absolute_url(root_url, link):
    logging.debug("INPUT link="+link)

    if link == "/":
        return link
    
    # Scarta i link che iniziano con "javascript:", "#" o altri schemi non HTTP
    if link.startswith(('javascript:', '#', 'mailto:', 'tel:')):
        return None

    # Rimuovi eventuali anchors dal link (e.g., "pagina.html#sezione1")
    link = link.split('#')[0]

    # Controlla se il link è assoluto o relativo
    if link.startswith('http') or link.startswith('https'):
        # Scarta i link esterni o quelli che non sono sotto la root_url
        if not link.startswith(root_url):
            return None
        return link
    else:
        # Assicuriamoci che il link inizi con "/"
        if not link.startswith("/"):
            link = "/" + link

        # Estrae le parti dell'URL root per confrontarle con il link relativo
        root_parts = root_url.split("/")[3:]  # [3:] per scartare "http:", "" e "dominio.com"
        
        # Divide il link relativo in parti
        link_parts = link.split("/")

        logging.debug(f"parts root={root_parts} e link={link_parts}")
        
        # Rimuovi le parti che corrispondono all'URL root dal link relativo
        for part in root_parts:
            if part in link_parts:
                link_parts.remove(part)
        
        # Ricostruisci il link relativo
        cleaned_link = "/".join(link_parts)
        cleaned_link = cleaned_link.replace("//", "/")
        cleaned_link = cleaned_link.replace("./", "/")
        cleaned_link = cleaned_link.replace("/./", "/")
        cleaned_link = cleaned_link.replace("../", "/")
        logging.debug("recostructed link ="+cleaned_link)

        # Converti i link relativi in link assoluti
        ret_link = urljoin(root_url, cleaned_link)
        logging.debug("absolute link="+ret_link)
        return ret_link
    
# Funzione ausiliaria ricorsiva per navigare nei link
def _crawl(root_url, base_url, url, current_depth:int, tag_name, attributes):
    valid_links = set()
    discarded_links = set()
    visited_links = set()

    if current_depth == 0:
        return valid_links, discarded_links, visited_links

    try:
        disable_warnings(InsecureRequestWarning)
        response = requests.get(url, verify=False)
        visited_links.add(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Estrai tutti i link dalla pagina
        page = soup.find(tag_name, **attributes) 
        if not page or type(page) != Tag :
            return set(), set(), set()
        
        l_links = page.find_all('a', href=True) # type: ignore
        logging.info("- Level " + str(current_depth) + " ---- ROOT = "+url+" *** NUM OF LINKS="+str(len(l_links))+" ***" )
        for a_tag in l_links:
            link = build_absolute_url(base_url, a_tag['href'])
            if not link:
                continue
                    
            # Controlla se il link è già stato visitato
            if not visited_links or link not in visited_links:
                if isValidLink(link, base_url):
                    valid_links.add(link)
                else:
                    discarded_links.add(link)
                ret_valid, ret_discarded, ret_visited = _crawl(root_url=root_url, base_url=base_url, url=link, current_depth=current_depth-1, tag_name=tag_name, attributes=attributes)
                if valid_links and ret_valid:
                    valid_links.update(ret_valid)
                if discarded_links and ret_discarded:
                    discarded_links.update(ret_discarded)
                if visited_links and ret_visited:
                    visited_links.update(ret_visited)

    except requests.RequestException as e:
        discarded_links.add(url)
        logging.info(f"Error fetching {url}: {e}")
    
    return valid_links, discarded_links, visited_links


