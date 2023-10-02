from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

csv_args={
    'delimiter': ';',
    'quotechar': '"',
    'fieldnames': ['INTERNAL ID','ICD10 CODE','DESCRIPTION']
}
loinc1_path = "./upload/icd10.csv"
print(f"loading file {loinc1_path}")

loader = CSVLoader(file_path=loinc1_path, csv_args=csv_args, source_column='ICD10 CODE')
data = loader.load()

print("creating index...")
faiss = FAISS.from_documents(documents=data, embedding=HuggingFaceEmbeddings())
#faiss = FAISS.load_local("kb/ICD10-3", embeddings=HuggingFaceEmbeddings())
#print("adding documents...")
#faiss.add_documents(documents=data)
print("saving index...")
faiss.save_local('kb/ICD10-3')
