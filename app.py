import streamlit as st
import backend
from pathlib import Path
import json
import utils
improt random
import datetime as dt

KEY_NO = 0
DATA_P = Path("./data")
st.title("Annotation Tool for relation extration")

savepath = st.text_input("Please enter the name of your output file (which will be saved to ./data) - DO NOT include any leading '/': ",
    value=f"train_triples_{dt.datetime.today().strftime('%Y-%m-%d')}.json",
    key=KEY_NO)
KEY_NO += 1

SAVEFILE = DATA_P.joinpath(savepath)
if not SAVEFILE.exists():
    SAVEFILE.parent.mkdir(parents=True, exist_ok=True)
    SAVEFILE.touch()

with SAVEFILE.open("r") as f:
    SAVE_STR = f.read()
    if SAVE_STR:
        SAVE_JSON = json.loads(SAVE_STR)
    else:
        SAVE_JSON = []
SAVE_DATA = {}

# load tokenizer
models = ["bert-base-cased", "bert-large-cased", "albert-xxlarge-v1"]
model_name = st.selectbox(label="Please select your pretrained model",
    options=models,
    key=KEY_NO)
KEY_NO += + 1

@st.cache(hash_funcs={backend.Tokenizer: str})
def load_tokenizer(model_name: str=model_name):
    return backend.Tokenizer(embedding=model_name)

tokenizer = load_tokenizer(model_name=model_name)

# default relation types
with DATA_P.joinpath("rel2idx.jsonc").open("r") as f:
    relation_types = json.loads(f.read())

with DATA_P.joinpath("ner2idx.jsonc").open("r") as f:
    ner_types = json.loads(f.read())


# process sentence
sentence_str = st.text_area("Enter your sentence:",
    key=KEY_NO)
KEY_NO += 1
tokens = tokenizer.tokenize(sentence_str)
SAVE_DATA["tokens"] = tokens


# display tokenized sentence
st.markdown("""Tokenized Sentence:\n""")
st.text(f"Tokens: {tokens}\nWriting to {SAVEFILE.as_posix()}")


# select NEs
max_entities = st.select_slider(
    label="slide this to increase the max # of entities",
    options=range(1, 21),
    value=3,
    key=KEY_NO)
KEY_NO += 1

named_entities = []
ne_names = {}

st.markdown("#### Annotate Entities:")
ent_col1, ent_col2 = st.columns(2)
i = 1
while i <= int(max_entities):
    ent = ent_col1.text_input(f"Enter named entity name #{i}:", 
        key=KEY_NO)
    KEY_NO += 1
    ent_tokens = tokenizer.tokenize(ent)
    _, ent_span = backend.list_overlap(tokens, ent_tokens)
    del _
    ent_type = ent_col2.selectbox(label=f"Select type of this entity (#{i})",
        options=ner_types.keys(),
        key=KEY_NO)
    KEY_NO += 1
    named_entities.append(
        {"type": ent_type,
        "start": ent_span[0],
        "end": ent_span[1]}
        )
    ne_names[ent] = i - 1
    i += 1
del i

SAVE_DATA["entities"] = named_entities

st.markdown("\n-------\n")
st.markdown("#### Annotate Relations:")

# select head and tail
max_relations = st.select_slider(
    label="slide this to increase the max # of relations",
    options=range(1, 10),
    value=3,
    key=KEY_NO)
KEY_NO += 1

rel_col1, rel_col2, rel_col3 = st.columns(3)
relations = []

i = 1
while i <= int(max_relations):

    head_str = rel_col1.selectbox(f"Select the #{i} head entity from the NEs ",
        options=ne_names.keys(),
        key=KEY_NO)
    head_idx = ne_names.get(head_str)
    KEY_NO += 1

    tail_str = rel_col2.selectbox("Select the tail entity from the NEs",
        options=ne_names.keys(),
        key=KEY_NO)
    tail_idx = ne_names.get(tail_str)
    KEY_NO += 1

    relation_type = rel_col3.selectbox(label=f"Select type of this relation (#{i})",
        options=relation_types.keys(),
        key=KEY_NO)
    KEY_NO += 1
    relations.append({"type": relation_type, "head": head_idx, "tail": tail_idx})
    i += 1

SAVE_DATA["relations"] = relations

j=random.randint(0,2000)
SAVE_DATA["orig_id"] = j

if isinstance(SAVE_JSON, list):    
    SAVE_JSON.append(SAVE_DATA)
elif isinstance(SAVE_JSON, dict):
    SAVE_JSON.update(SAVE_DATA)
else:
    raise TypeError(f"unrecognized file type {SAVEFILE.as_posix()}")

st.text("Please click save to save your inputs")
KEY_NO += 1
SAVE = st.button("Save", key=KEY_NO)
KEY_NO += 1

if SAVE:
    st.text(f"Saving to {SAVEFILE.as_posix()}...")
    with SAVEFILE.open("w") as f:
        save_str = json.dumps(SAVE_JSON)
        st.text(save_str)
        f.write(save_str)

with SAVEFILE.open("r") as f:
    st.download_button("Download *.json file", f)
