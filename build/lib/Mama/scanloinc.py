from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['LOINC_NUM','COMPONENT','PROPERTY','TIME_ASPCT','SYSTEM','SCALE_TYP','METHOD_TYP','CLASS','CLASSTYPE','LONG_COMMON_NAME','SHORTNAME','EXTERNAL_COPYRIGHT_NOTICE','STATUS','VersionFirstReleased','VersionLastChanged']
}
loinc1_path = "./upload/Loinc.csv"
print(f"loading file {loinc1_path}")

loader = CSVLoader(file_path=loinc1_path, csv_args=csv_args, source_column='LOINC_NUM')
data = loader.load()

#faiss = FAISS.from_documents(documents=data, embedding=HuggingFaceEmbeddings())
print("loading index...")
faiss = FAISS.load_local("kb/loinc", embeddings=HuggingFaceEmbeddings())
print("adding documents...")
faiss.add_documents(documents=data)
print("saving index...")
faiss.save_local('kb/loinc')


