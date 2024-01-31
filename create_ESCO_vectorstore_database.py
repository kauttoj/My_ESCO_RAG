import os
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd

# code to precompute persisten vectorstore databases to be used during retrieval. Databases are stored on disk.
# YOU MUST HAVE "OPENAI_API_KEY" environmental variable set!

OPENAI_MODEL ="text-embedding-ada-002" # advanced embedding model (as of January 2024)
HUGGINFACE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # simple free embedding model
DATABASE_DIR = os.getcwd() + os.sep + 'ESCO_database_' # output paths
SOURCE_DATA_PATH = 'C:\Code\ESCO_GPT\data\ESCO dataset - v1.1.1 - classification - {0} - csv'  # path to ESCO data, we add here language IDs (fi and en)

# create embedding functions
openai_ef = OpenAIEmbeddings(
    model_name="text-embedding-ada-002",
    api_key=os.getenv('OPENAI_API_KEY'),
)
huggingface_ef = SentenceTransformerEmbeddings(model_name=HUGGINFACE_MODEL)

for model in ['openai','huggingface']:
    if model == 'openai':
        embed_model=openai_ef
    else:
        embed_model = huggingface_ef

    for lang in ['en','fi']:
        DATABASE_DIR1=DATABASE_DIR+model+'_'+lang
        collection_name = 'ESCO_' + model + '_' + lang
        try:
            vectorstore = Chroma(persist_directory=DATABASE_DIR1,embedding_function=embed_model,collection_name=collection_name)
            assert vectorstore._collection.count() > 10, 'Empty database!'
            print("Old database loaded and ready")
        except:
            print("Creating a new database, this might cost you!")
            database = pd.read_csv(SOURCE_DATA_PATH + '\skills_{0}.csv'.format(lang))
            database = database.loc[database.skillType == 'skill/competence']
            database = database.loc[(database.description.map(lambda x:len(x))>30) & (database.preferredLabel.map(lambda x:len(x))>5)]
            assert len(database)>100,'Failed to read!'
            skills = [row[1]['preferredLabel'] + ' | ' + row[1]['description'] for row in database.iterrows()]
            metadatas1 = [{'altLabels':row[1]['altLabels'],'modifiedDate':row[1]['modifiedDate'],'type':'skill/competence','url':row[1]['conceptUri']} for row in database.iterrows()]
            assert len(skills) > 100,'Empty database!'

            database = pd.read_csv(SOURCE_DATA_PATH + '\occupations_{0}.csv'.format(lang))
            database = database.loc[database.conceptType == 'Occupation']
            database = database.loc[(database.description.map(lambda x:len(x))>30) & (database.preferredLabel.map(lambda x:len(x))>2)]
            occupations = [row[1]['preferredLabel'] + ' | ' + row[1]['description'] for row in database.iterrows()]
            metadatas2 = [{'modifiedDate':row[1]['modifiedDate'],'type':'occupation','url':row[1]['conceptUri']} for row in database.iterrows()]
            assert len(occupations) > 10,'Empty database!'

            documents = skills + occupations
            metadatas = metadatas1 + metadatas2

            assert len(documents)==len(metadatas),'bad size!'

            print('embedding %i documents' % len(documents))
            vectorstore = Chroma.from_texts(
                texts=documents,
                collection_name=collection_name,
                embedding=embed_model,
                metadatas=metadatas,
                persist_directory = DATABASE_DIR1,
            )
            vectorstore.persist()
            assert vectorstore._collection.count() > 10, 'Empty database!'
            print('new database size is %i documents!' % vectorstore._collection.count())

            print('database created and saved!')

print('all done!')
