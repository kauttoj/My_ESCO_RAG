import tkinter as tk
import json
import requests
from urllib.request import urlopen
import ssl
import urllib
import time
import os

# This is a test code for HEADAI model to convert text into skill terms. You can choose either new or old API. You need tokens to run these.

# You must have HEADAI token in your environmental variables as HEADAI_TOKEN!
LANGUAGE = 'en' # en  # choose language, not sure if matters...
USE_OLD_API = 0  # use old of new API
ONTOLOGY = 'esco' # headai # which ontology to use, options esco and headai

def on_submit():
    teksti = input_textbox.get("1.0", tk.END).strip()
    if len(teksti)<5:
        print('processing example text')
        teksti = 'Haemme nyt Diktamenin kasvavaan joukkoon erikoissairaanhoidon tekstinkäsittelijöitä kokoaikaiseen työsuhteeseen. Meillä voit työskennellä toimistoaikojen puitteissa tai kaksivuorotyössä. Tekstinkäsittelijänä tehtäväsi on muuttaa lääkärin sanelema potilaskertomus kirjalliseen muotoon ja kirjata teksti potilasdokumenteiksi erilaisiin potilasjärjestelmiin. Käytännössä vastaat sinulle määritettyjen asiakkaiden saneluiden purkamisesta laatustandardiemme mukaisesti. Työsi on tärkeä osa terveydenhuollon kokonaisuutta, sillä oikea-aikainen potilasdokumentointi on kriittinen osa suomalaista terveydenhuoltojärjestelmää. Työsi ansiosta lääkärit voivat tehdä työnsä tehokkaammin ja vapauttaa aikaansa ihmisten auttamiseen. Sanelunpurkua ja potilasdokumentointia ei opeteta koulussa, joten perehdytämme sinut työhösi Helsingissä sijaitsevalla toimistollamme tai etänä videoyhteyksien välityksellä. Kokeneet sanelunpurun ammattilaiset huolehtivat, että opit ammatissa vaadittavat tiedot ja taidot pikavauhtia. Perehdytyksen jälkeen sinusta tulee oman asiakkuutesi asiantuntija, ja hallitset hyvin lääketieteen sanaston sekä muutamia potilastietojärjestelmiä. Voit työskennellä etänä mistä päin Suomea tahansa tai toimistollamme Helsingissä, Lahdessa, Kotkassa, Hämeenlinnassa tai Kouvolassa. Erikoissairaanhoidon tekstinkäsittelijän tehtävään odotamme aiempaa kokemusta alalta, eli sinulla tulisi olla sote-alan taustaa ja kyky hallita lääketieteellistä sanastoa. Jotta menestyt tässä työssä, sinulla on erinomainen kirjallinen suomen kielen taito ja kyky oppia uusia asioita nopeasti. Lisäksi käytät tietokonetta sujuvasti ja omaat hyvät tiedonhakutaidot. Bonusta saat ruotsin ja englannin kielen taidosta sekä yrittäjähenkisestä palveluasenteesta. Olemme erittäin innoissamme, jos hallitset kymmensormijärjestelmän tai potilastietojärjestelmien käyttämisen.'
        input_textbox.insert("1.0",teksti)

    if USE_OLD_API:
        URL = "https://3amkapi.headai.com/microcompetencies?action=text_to_skills&"
        TOKEN = os.getenv('HEADAI_TOKEN_OLD')
    else:
        URL = "https://megatron.headai.com/TextToKeywords?"
        TOKEN = os.getenv('HEADAI_TOKEN')


    TOKEN_str = "language=%s&ontology=%s&token=%s&" % (LANGUAGE,ONTOLOGY,TOKEN) # set language and ontology

    ssl._create_default_https_context = ssl._create_unverified_context
    url = URL + TOKEN_str + "text=" + urllib.parse.quote(teksti)

    # wait for results
    headai_words = {None:None}
    done=False
    while 1:
        response = requests.get(url=url, timeout=15, data={}).json()
        time.sleep(2)
        try:
            if USE_OLD_API:
                headai_words = response['data']
            else:
                data_json = json.loads(urlopen(response['location'], timeout=5).read())
                headai_words = {x['concept']: {'weight': x['weight'], 'count': x['count']} for x in data_json['skills']}
                headai_words = list(headai_words.keys())
            done=True
        except:
            print('waiting for results...')
        if done:
            break

    print('Result: %s' % str(headai_words))
    create_word_boxes(headai_words)

def create_word_boxes(terms):
    frame = word_box_frame
    for k,term in enumerate(terms):
        button = tk.Button(frame, text=term, borderwidth=2, relief="solid")
        button.pack(side=tk.LEFT, padx=2, pady=2)
    rearrange_buttons(frame)
def rearrange_buttons(event):
    frame = word_box_frame
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

def clear_all():
    input_textbox.delete("1.0", tk.END)
    frame = word_box_frame
    buttons = [child for child in frame.winfo_children() if isinstance(child, tk.Button)]
    for button in buttons:
        button.destroy()

root = tk.Tk()

# Set window title and size
root.title("HEADAI testing")
root.geometry("1300x1000")

# Configure rows and columns
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, minsize=30)  # Fixed width for the second column

# Invisible frame to maintain the width of the second column
fixed_width_frame = tk.Frame(root, width=30)
fixed_width_frame.grid(row=0, column=1, rowspan=2)
fixed_width_frame.grid_propagate(False)  # Prevents the frame from resizing

# First button in row 1, column 1
input_textbox = tk.Text(root, bg="lightblue", height=15, border=True)
input_textbox.grid(row=0, column=0, sticky="nsew")

# Frame for row 1-2, column 2
col2_frame = tk.Frame(root, bg="azure")
col2_frame.grid(row=0, column=1, rowspan=1, sticky="nsew")

# Spacer frame at the top for centering
top_spacer = tk.Frame(col2_frame, height=10, bg="azure")
top_spacer.pack(side="top", fill="x", expand=True)

submit_button = tk.Button(col2_frame, text="Submit", command=on_submit, width=20)
submit_button.pack(side="top", pady=(10, 0))  # Adjust padding as needed
clear_button = tk.Button(col2_frame, text="Clear", command=clear_all, width=20)
clear_button.pack(side="top", pady=(10, 0))  # Adjust padding as needed

# Spacer frame at the bottom for centering
bottom_spacer = tk.Frame(col2_frame, height=10, bg="azure")
bottom_spacer.pack(side="bottom", fill="x", expand=True)

word_box_frame = tk.Frame(root, bg="misty rose", border=True)
word_box_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
word_box_frame.bind("<Configure>", lambda event: rearrange_buttons(event))
root.grid_rowconfigure(1, weight=1)

root.mainloop()