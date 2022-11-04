# Small tool for annotating text

## Installation
- clone this repository then run `$ conda env create -f environment.yml`
- Run `$ conda activate annotation_env` to activate the environment

## Running the app
- If you are running this locally, run `streamlit run app.py` to start the annotation app
- If you are running on a server, you need to listen to the same channel through ssh. 
Run `ssh -L xxxx:localhost:xxxx -i <path/to/your/key/file> <yourusername>@<yourserveraddress> -p <yourport#>`, then access the address from your browser

## Modifying the relation types
- Made modifications in `./data/ner2idx.jsonc` and `./data/rel2idx.jsonc` to modify the relation and NE types
