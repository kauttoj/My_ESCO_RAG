import tkinter as tk
import json
import numpy as np
import os
import re
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.chroma import Chroma
import tiktoken

from langchain.globals import set_debug
set_debug(False) # set True to get more details of process

# This is a test code for a simple RAG system to convert text into skill and occupation suggestions. This simple RAG uses LLM embeddings as vector database (retrieval) and LLM chat model (augmented generation) to pick most relevant skills. You need OpenAI token to use GPT-3.5 and GPT-4.
# Currently retrieval can be made either with OpenAI model or local Hugginface model. Generation can be only made via OpenAI.

# !!! Note: All API calls to OpenAI will cost you a little. Check their current pricing and monitor your usage !!!

# LLM_MODEL is used in generating final answer, must be some "advanced" model. YOU MUST HAVE "OPENAI_API_KEY" environmental variable set!
LLM_MODEL =  'gpt-3.5-turbo-1106' # 'gpt-4-1106-preview'
VECTORSTORE_MAX_RETRIEVED = 25 # Max number of documents to retrieve from vectorstore. Don't go over context window
VECTORSTORE_MAX_SUGGESTED = [20,10] # for [skills, occupations] how many potential items to suggest from vectorstore
LLM_MAX_PICKS = [15,5] # for [skills, occupations] how many items LLM must pick from retrieved options

# paths to precomputed databases for dedicated languages
LANGUAGE_SEL = 'English' # can do also Finnish for multilingual models
#LANGUAGE_SEL = 'Finnish' # best for finnish content
if 1:
    print('Using OpenAI embeddings for retrieval')
    DATABASE_DIRS = {'en':os.getcwd() + os.sep + 'ESCO_database_openai_en','fi':os.getcwd() + os.sep + 'ESCO_database_openai_fi'}
    embedding_ef = OpenAIEmbeddings(
        model_name="text-embedding-ada-002",
        api_key=os.getenv('OPENAI_API_KEY'),
    )
else:
    print('Using Huggingface model for retrieval')
    DATABASE_DIRS = {'en': os.getcwd() + os.sep + 'ESCO_database_huggingface_en', 'fi': os.getcwd() + os.sep + 'ESCO_database_huggingface_fi'}
    embedding_ef= SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# template for getting final answer from LLM
MY_PROMPT_TEMPLATE = """
### TASK
You are doing work skills or occupation matching in {language}. You are given a list of items from which you need to pick those that match best with a given text content. 

### RULES OF TASK
Strictly Follow these guidelines when answering:
1. You can only choose items from a given list below. DO NOT INVENT any new items not in list.
2. For chosen skills or occupations, give a scoring of LOW, MEDIUM or HIGH depending how well each item matches the content of the user text.
3. Give your response in JSON format {{'items':[item1, item2,...],'relevances':['LOW','MEDIUM',...]}}

### BEGIN USER TEXT 

{question}

### END USER TEXT

The following list contains skills or occupations with their descriptions. Each ITEM and DESCRIPTION are separated by a pipe "|" symbol. Only include items in your response.
Think carefully and choose maximum {LLM_MAX_PICKS} items from the list, remember RULES OF TASK.

### BEGIN ITEM DESCRIPTIONS

{context}

### END ITEM DESCRIPTIONS

Your answer:
"""

encoding = tiktoken.encoding_for_model(LLM_MODEL)
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens

def format_docs(docs) -> str:
    return "\n\n".join([d.page_content for d in docs])

class Computations:

    def __init__(self, gui_manager):
        self.gui_manager = gui_manager

        print("Loading vectorstore")
        collection_name = DATABASE_DIRS[LANGUAGE_SEL[0:2].lower()].split(os.sep)[-1].replace('database_','')
        vectorstore = Chroma(persist_directory=DATABASE_DIRS[LANGUAGE_SEL[0:2].lower()],collection_name=collection_name,embedding_function=embedding_ef)
        assert vectorstore._collection.count() > 10, 'Empty database!'

        print('Vectorstore size is %i' % (vectorstore._collection.count()))

        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": VECTORSTORE_MAX_RETRIEVED}
        )

        # prompt GPT with Langchain
        self.prompt = ChatPromptTemplate.from_template(MY_PROMPT_TEMPLATE)
        self.model = ChatOpenAI(model=LLM_MODEL, temperature=0.01, max_tokens=4096)

        ## not using this as we want more control
        #self.chain = (
        #        {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
        #        | self.prompt
        #        | self.model
        #        | StrOutputParser()
        #)

        print('RAG setup finished!')

    def get_similar_words(self,docs,doc_type,n_suggest):
        used_docs = set([x['document'] for x in docs])
        n_old = len(used_docs)
        n_new = 2*int(np.ceil(n_suggest / n_old))
        new_docs = []
        for current_doc in docs:
            suggested_docs = self.retriever.vectorstore._collection.query(query_embeddings=[current_doc['embedding']], n_results=n_old + n_new,include=['embeddings','documents','metadatas'],where={"type": doc_type})
            z = suggested_docs['documents'][0]
            cands = [{'document':z[k],'embedding':suggested_docs['embeddings'][0][k]} for k in range(len(z)) if (z[k] not in used_docs)]
            new_docs.append(cands)
            used_docs.update([x['document'] for x in cands])
        round=-1
        final_docs = []
        # next we pick suggestions from candidates such that each terms provides something
        while 1:
            round+=1
            is_empty = True
            for cands in new_docs:
                if len(cands)>round:
                    is_empty=False
                    if len(final_docs) < n_suggest:
                        final_docs.append(cands[round])
                    else:
                        break
                else:
                    continue
            if is_empty or len(final_docs) == n_suggest:
                break
        for k,x in enumerate(final_docs):
            final_docs[k]['skill'] = x['document'].split('|')[0].strip()

        return final_docs

    def clean_docs(self,docs):
        return [x.split('|')[0].strip() for x in docs]

    def process_submit(self, input_text):
        # Process the input text as needed
        # embed reviews
        question_vec = self.retriever.vectorstore._embedding_function.embed_query(input_text)

        def process_query(doc_type,MAX_PICKS):
             # NLP model call 1 (possibly API)
            retrieved_docs = self.retriever.vectorstore._collection.query(query_embeddings=[question_vec], n_results=VECTORSTORE_MAX_RETRIEVED,include=['embeddings','documents','metadatas'],where={"type": doc_type})

            context = [x.replace('\n', ' ').replace('\r', '') for x in retrieved_docs['documents'][0]]
            context = [x for x in context if len(x)>4]
            context = '\n\n'.join(context)

            # use chain with precomputed context field, we only insert question
            precomputed_context = MY_PROMPT_TEMPLATE.replace('{language}',LANGUAGE_SEL).replace('{LLM_MAX_PICKS}', str(MAX_PICKS)).replace('{context}',context)
            n_tokens = num_tokens_from_string(precomputed_context)
            print('Prompt size is %i tokens' % n_tokens)
            if n_tokens>self.model.max_tokens:
                print('!!!!! Number of tokens beyond typical GPT-3.5 model %i > %i !!!!!!!' % (n_tokens,self.model.max_tokens))

            precomputed_chain = (
                    {"question": RunnablePassthrough()}
                    | ChatPromptTemplate.from_template(precomputed_context)
                    | self.model
                    | StrOutputParser()
            )
            print("Prompting LLM with %i best-matching documents" % VECTORSTORE_MAX_RETRIEVED)
            answer = precomputed_chain.invoke(input_text) # NLP model call 2 (possibly API)
            #answer = self.chain.invoke(input_text)  # this uses retriever as part of chain, but does not return documents nor embeddings

            output_text = "\n".join(['%i: %s' % (k + 1, x) for k, x in enumerate(context.replace('\n\n', '\n').split('\n'))])
            print('ALL RETRIEVED DOCUMENTS:\n%s' % output_text)
            #self.gui_manager.update_output_text(output_text)

            # cleaning and verifying proper JSON formatting
            answer = answer.replace('\n','').replace('\t','')
            inds = (answer.find('{'),answer.rfind('}'))
            assert inds[0]>-1 and inds[1]>-1,'Response not in valid JSON format!'
            answer = answer[inds[0]:(inds[1]+1)]
            try:
                answer = json.loads(answer)
            except:
                raise(Exception('Response not in valid JSON format!'))

            # verify and try to fix errors in response
            assert ('items' in answer) and ('relevances' in answer), 'Response not in correct format!'
            di = len(answer['relevances'])-len(answer['items'])
            if di>0:
                answer['relevances'] = answer['relevances'][0:len(answer['items'])]
            if di<0:
                answer['relevances'] = answer['relevances'] + ['LOW']*(-di)
            # remove duplicates if any
            idx = np.where(pd.DataFrame(answer['items']).duplicated(keep=False))[0]
            answer['relevances'] = [x.strip() for k,x in enumerate(answer['relevances']) if (len(idx)==0 or k not in idx)]
            answer['items'] = [x.strip() for k, x in enumerate(answer['items']) if (len(idx)==0 or k not in idx)]
            answer = {answer['items'][i]:answer['relevances'][i] for i in range(len(answer['items']))}

            # make structured terms
            selected_docs = []
            words_set = set(answer.keys())
            relevance_mapping = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
            for k,x in enumerate(retrieved_docs['documents'][0]):
                for y in words_set:
                    skill = x.split('|')[0]
                    if y in skill:
                        selected_docs.append({'embedding':retrieved_docs['embeddings'][0][k],'document':x,'relevance': answer[y]})
                        selected_docs[-1]['relevance'] = relevance_mapping[selected_docs[-1]['relevance']]
                        selected_docs[-1]['skill'] = skill.strip()
                        words_set.remove(y)
                        break
            if len(words_set)>0:
                print('!!! Possible hallucination detected! Following items were not found in context: %s' % str(words_set))
            # in case we don't have enough LLM picks, use vectorstore picks
            if len(selected_docs)<4:
                print('Adding 3 best vectorstore suggestions to existing %i LLM suggestions' % (len(selected_docs)))
                selected_docs+=[{'embedding':retrieved_docs['embeddings'][0][k],'document':retrieved_docs['documents'][0][k],'relevance': 2,'skill':retrieved_docs['documents'][0][k].split('|')[0].strip()} for k in range(3)]

            return selected_docs

        print('\nExecuting RAG for skills')
        selected_docs = process_query('skill/competence',LLM_MAX_PICKS[0])
        terms = sorted(selected_docs, key=lambda x: x['relevance'], reverse=True)
        self.gui_manager.create_word_boxes(1,terms) # fill box 1
        suggested_docs = self.get_similar_words(selected_docs,'skill/competence',VECTORSTORE_MAX_SUGGESTED[0])
        assert len(set([x['skill'] for x in selected_docs]).intersection([x['skill'] for x in suggested_docs]))==0,'selected and suggested have overlap!'
        self.gui_manager.create_word_boxes(2,suggested_docs) # fill box 2

        print('\nExecuting RAG for occupations')
        selected_docs = process_query('occupation',LLM_MAX_PICKS[1])
        terms = sorted(selected_docs, key=lambda x: x['relevance'], reverse=True)
        self.gui_manager.create_word_boxes(3,terms) # fill box 3
        suggested_docs = self.get_similar_words(selected_docs,'occupation',VECTORSTORE_MAX_SUGGESTED[1])
        assert len(set([x['skill'] for x in selected_docs]).intersection([x['skill'] for x in suggested_docs])) == 0, 'selected and suggested have overlap!'
        self.gui_manager.create_word_boxes(4,suggested_docs) # fill box 4

    def process_clear(self):
        self.gui_manager.clear_texts()

    def get_word_list(self, type='normal'):
        if type == 'normal':
            return ["word_%i" % (i+1) for i in range(30)]
        else:
            return ["second_word_%i" % (i+1) for i in range(30)]

class GuiManager:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("My ESCO RAG")
        self.root.geometry("1200x900")

        self.computations = None
        self.setup_widgets()

    def copy(self,event):
        pass
        #self.root.clipboard_clear()
        #self.text = event.widget.get("sel.first", "sel.last")
        #self.root.clipboard_append(self.text)

    def paste(self,event):
        pass
        #try:
        #    text = self.root.clipboard_get()
        #    event.widget.insert("insert", text)
        #except tk.TclError:
        #    pass  # No text in clipboard

    def set_computations(self, computations):
        self.computations = computations

    def on_clicked_current(self,button):
        #self.add_to_second_word_box(button.cget('text'))
        button.destroy()
        self.rearrange_buttons(self.word_box_frame)
        documents = [{'document':child.document,'skill':child.cget('text'),'embedding':child.embedding} for child in self.word_box_frame.winfo_children() if isinstance(child, tk.Button)]
        new_suggested = self.computations.get_similar_words(documents,'skill/competence',VECTORSTORE_MAX_SUGGESTED[0])
        self.clear_words(2)
        self.create_word_boxes(2,new_suggested)
        self.rearrange_buttons(self.second_word_box_frame)

    def on_clicked_suggested(self,button):
        self.add_to_first_word_box({'document': button.document, 'skill': button.cget('text'), 'embedding': button.embedding})
        button.destroy()
        self.rearrange_buttons(self.word_box_frame)
        documents = [{'document': child.document, 'skill': child.cget('text'), 'embedding': child.embedding} for child in self.word_box_frame.winfo_children() if isinstance(child, tk.Button)]
        new_suggested = self.computations.get_similar_words(documents,'skill/competence',VECTORSTORE_MAX_SUGGESTED[0])
        self.clear_words(2)
        self.create_word_boxes(2,new_suggested)
        self.rearrange_buttons(self.second_word_box_frame)

    def on_clicked_current_occu(self,button):
        #self.add_to_second_word_box(button.cget('text'))
        button.destroy()
        self.rearrange_buttons(self.third_word_box_frame)
        documents = [{'document':child.document,'skill':child.cget('text'),'embedding':child.embedding} for child in self.third_word_box_frame.winfo_children() if isinstance(child, tk.Button)]
        new_suggested = self.computations.get_similar_words(documents,'occupation',VECTORSTORE_MAX_SUGGESTED[1])
        self.clear_words(4)
        self.create_word_boxes(4,new_suggested)
        self.rearrange_buttons(self.fourth_word_box_frame)

    def on_clicked_suggested_occu(self,button):
        self.add_to_first_word_box_occu({'document': button.document, 'skill': button.cget('text'), 'embedding': button.embedding})
        button.destroy()
        self.rearrange_buttons(self.third_word_box_frame)
        documents = [{'document': child.document, 'skill': child.cget('text'), 'embedding': child.embedding} for child in self.third_word_box_frame.winfo_children() if isinstance(child, tk.Button)]
        new_suggested = self.computations.get_similar_words(documents,'occupation',VECTORSTORE_MAX_SUGGESTED[1])
        self.clear_words(4)
        self.create_word_boxes(4,new_suggested)
        self.rearrange_buttons(self.fourth_word_box_frame)

    def setup_widgets(self):

        # Configure rows and columns
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, minsize=30)  # Fixed width for the second column

        # Invisible frame to maintain the width of the second column
        fixed_width_frame = tk.Frame(self.root, width=30)
        fixed_width_frame.grid(row=0, column=1, rowspan=2)
        fixed_width_frame.grid_propagate(False)  # Prevents the frame from resizing

        # First button in row 1, column 1
        self.input_textbox = tk.Text(self.root, bg="lightblue", height=10, border=True)
        self.input_textbox.grid(row=0, column=0, sticky="nsew")

        # Second button in row 2, column 1
        #self.output_textbox = tk.Text(self.root, bg="lightgreen", height=5, border=True)
        #self.output_textbox.grid(row=1, column=0, sticky="nsew")

        self.input_textbox.bind("<Button-3><ButtonRelease-3>", lambda event: self.input_textbox.tag_add("sel", "@%s,%s" % (event.x, event.y)))
        self.input_textbox.bind("<Control-c>", self.copy)
        self.input_textbox.bind("<Control-v>", self.paste)

        #self.output_textbox.bind("<Button-3><ButtonRelease-3>", lambda event: self.output_textbox.tag_add("sel", "@%s,%s" % (event.x, event.y)))
        #self.output_textbox.bind("<Control-c>", self.copy)

        # Frame for row 1-2, column 2
        col2_frame = tk.Frame(self.root, bg="azure")
        col2_frame.grid(row=0, column=1, rowspan=1, sticky="nsew")

        # Spacer frame at the top for centering
        top_spacer = tk.Frame(col2_frame, height=10, bg="azure")
        top_spacer.pack(side="top", fill="x", expand=True)

        submit_button = tk.Button(col2_frame, text="Submit", command=self.on_submit, width=20)
        submit_button.pack(side="top", pady=(10, 0))  # Adjust padding as needed
        clear_button = tk.Button(col2_frame, text="Clear", command=self.clear_all, width=20)
        clear_button.pack(side="top", pady=(10, 0))  # Adjust padding as needed

        # Entry in col2_frame
        #entry2_col2 = tk.Entry(col2_frame, width=20, bg="lightyellow")
        #entry2_col2.pack(side="top", pady=(10, 0))  # Adjust padding as needed

        # Spacer frame at the bottom for centering
        bottom_spacer = tk.Frame(col2_frame, height=10, bg="azure")
        bottom_spacer.pack(side="bottom", fill="x", expand=True)

        self.word_box_frame = tk.Frame(self.root, bg="misty rose",border=True)
        self.word_box_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.word_box_frame.bind("<Configure>", lambda event: self.on_frame_configure(event))
        self.root.grid_rowconfigure(1, weight=1)

        self.second_word_box_frame = tk.Frame(self.root, bg="misty rose",border=True)
        self.second_word_box_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.second_word_box_frame.bind("<Configure>", lambda event: self.on_frame_configure(event))
        self.root.grid_rowconfigure(2, weight=1)

        self.third_word_box_frame = tk.Frame(self.root, bg="light cyan",border=True)
        self.third_word_box_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.third_word_box_frame.bind("<Configure>", lambda event: self.on_frame_configure(event))
        self.root.grid_rowconfigure(3, weight=1)

        self.fourth_word_box_frame = tk.Frame(self.root, bg="light cyan",border=True)
        self.fourth_word_box_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.fourth_word_box_frame.bind("<Configure>", lambda event: self.on_frame_configure(event))
        self.root.grid_rowconfigure(4, weight=1)

    def add_to_second_word_box(self,term):
        frame = self.second_word_box_frame
        button = tk.Button(frame, text=term['skill'], borderwidth=2, relief="solid")
        button.embedding = term['embedding']
        button.document = term['document']
        button.config(bg='SlateGray1',command=(lambda x=button: self.on_clicked_suggested(x)))
        button.pack(side=tk.LEFT, padx=2, pady=2)

    def add_to_first_word_box(self,term):
        frame = self.word_box_frame
        button = tk.Button(frame, text=term['skill'], borderwidth=2, relief="solid")
        button.embedding = term['embedding']
        button.document = term['document']
        button.config(bg='SlateGray1',command=(lambda x=button: self.on_clicked_current(x)))
        button.pack(side=tk.LEFT, padx=2, pady=2)

    def add_to_second_word_box_occu(self,term):
        frame = self.fourth_word_box_frame
        button = tk.Button(frame, text=term['skill'], borderwidth=2, relief="solid")
        button.embedding = term['embedding']
        button.document = term['document']
        button.config(bg='SlateGray1',command=(lambda x=button: self.on_clicked_suggested_occu(x)))
        button.pack(side=tk.LEFT, padx=2, pady=2)

    def add_to_first_word_box_occu(self,term):
        frame = self.third_word_box_frame
        button = tk.Button(frame, text=term['skill'], borderwidth=2, relief="solid")
        button.embedding = term['embedding']
        button.document = term['document']
        button.config(bg='SlateGray1',command=(lambda x=button: self.on_clicked_current_occu(x)))
        button.pack(side=tk.LEFT, padx=2, pady=2)

    def create_word_boxes(self,box_num,terms):
        if box_num==1:
            frame = self.word_box_frame
            click_fun = self.on_clicked_current
        elif box_num==2:
            frame = self.second_word_box_frame
            click_fun = self.on_clicked_suggested
        elif box_num == 3:
            frame = self.third_word_box_frame
            click_fun = self.on_clicked_current_occu
        elif box_num == 4:
            frame = self.fourth_word_box_frame
            click_fun = self.on_clicked_suggested_occu
        else:
            raise(Exception('Bad box number!'))

        for k,term in enumerate(terms):
            button = tk.Button(frame, text=term['skill'], borderwidth=2, relief="solid")
            button.config(command= (lambda x=button: click_fun(x))) # bg=BUTTON_COLOR_fun(term['relevance'])
            button.embedding = term['embedding']
            button.document = term['document']
            button.pack(side=tk.LEFT, padx=2, pady=2)
        self.rearrange_buttons(frame)

    def on_submit(self):
        input_text = self.input_textbox.get("1.0", tk.END).strip()
        if len(input_text)==0: # for testing, use default string if left empty
            if LANGUAGE_SEL=='English':
                input_text='I have worked last 5 years as an contractor that sells and installs heat pumps for consumers. I have engineering degree and know lots of refrigenerators, and technology of heating and cooling systems. I have licence to do electrical jobs.'
            else:
                input_text='Haemme nyt Diktamenin kasvavaan joukkoon erikoissairaanhoidon tekstinkäsittelijöitä kokoaikaiseen työsuhteeseen. Meillä voit työskennellä toimistoaikojen puitteissa tai kaksivuorotyössä. Tekstinkäsittelijänä tehtäväsi on muuttaa lääkärin sanelema potilaskertomus kirjalliseen muotoon ja kirjata teksti potilasdokumenteiksi erilaisiin potilasjärjestelmiin. Käytännössä vastaat sinulle määritettyjen asiakkaiden saneluiden purkamisesta laatustandardiemme mukaisesti. Työsi on tärkeä osa terveydenhuollon kokonaisuutta, sillä oikea-aikainen potilasdokumentointi on kriittinen osa suomalaista terveydenhuoltojärjestelmää. Työsi ansiosta lääkärit voivat tehdä työnsä tehokkaammin ja vapauttaa aikaansa ihmisten auttamiseen. Sanelunpurkua ja potilasdokumentointia ei opeteta koulussa, joten perehdytämme sinut työhösi Helsingissä sijaitsevalla toimistollamme tai etänä videoyhteyksien välityksellä. Kokeneet sanelunpurun ammattilaiset huolehtivat, että opit ammatissa vaadittavat tiedot ja taidot pikavauhtia. Perehdytyksen jälkeen sinusta tulee oman asiakkuutesi asiantuntija, ja hallitset hyvin lääketieteen sanaston sekä muutamia potilastietojärjestelmiä. Voit työskennellä etänä mistä päin Suomea tahansa tai toimistollamme Helsingissä, Lahdessa, Kotkassa, Hämeenlinnassa tai Kouvolassa. Erikoissairaanhoidon tekstinkäsittelijän tehtävään odotamme aiempaa kokemusta alalta, eli sinulla tulisi olla sote-alan taustaa ja kyky hallita lääketieteellistä sanastoa. Jotta menestyt tässä työssä, sinulla on erinomainen kirjallinen suomen kielen taito ja kyky oppia uusia asioita nopeasti. Lisäksi käytät tietokonetta sujuvasti ja omaat hyvät tiedonhakutaidot. Bonusta saat ruotsin ja englannin kielen taidosta sekä yrittäjähenkisestä palveluasenteesta. Olemme erittäin innoissamme, jos hallitset kymmensormijärjestelmän tai potilastietojärjestelmien käyttämisen.'
            self.input_textbox.insert("1.0", input_text)
        self.computations.process_submit(input_text)

    def clear_all(self):
        self.clear_input()
        self.clear_words(1)
        self.clear_words(2)
        self.clear_words(3)
        self.clear_words(4)

    def clear_input(self):
        self.input_textbox.delete("1.0", tk.END)

    def clear_words(self,box_num):
        if box_num==1:
            frame = self.word_box_frame
        elif box_num==2:
            frame = self.second_word_box_frame
        elif box_num == 3:
            frame = self.third_word_box_frame
        elif box_num == 4:
            frame = self.fourth_word_box_frame
        else:
            raise(Exception('Bad box number!'))
        buttons = [child for child in frame.winfo_children() if isinstance(child, tk.Button)]
        for button in buttons:
            button.destroy()
    def rearrange_buttons(self,frame):
        x = y = 0
        buttons = [child for child in frame.winfo_children() if isinstance(child, tk.Button)]
        for button in buttons:
            frame.update_idletasks()  # Update the geometry of the frame
            button_width = button.winfo_width()
            frame_width = frame.winfo_width()
            if x + button_width > frame_width:
                x = 0
                y += button.winfo_height()
            button.place(x=x, y=y)
            x += button_width

    def on_frame_configure(self,event):
        self.rearrange_buttons(event.widget)
    def mainloop(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui_manager = GuiManager() # create GUI
    computations = Computations(gui_manager) # create computing object
    gui_manager.set_computations(computations)
    gui_manager.mainloop() # start GUI loop
