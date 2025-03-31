import yaml

from transformers import EsmTokenizer


# Function to load yaml configuration file
def load_config(config_fname):
    with open(config_fname) as file:
        config = yaml.safe_load(file)
    return config

def get_foldseek_tokenizer():
    all_fs_tokens = ['<cls>', '<pad>', '<eos>', '<unk>', 'p', 'y', 'n', 'w', 'r', 'q', 'h', 'g', 
                        'd', 'l', 'v', 't', 'm', 'f', 's', 'a', 'e', 'i', 'k', 'c', '#']
    with open("vocab.txt", "w") as f:
        f.write("\n".join(all_fs_tokens))
    fs_tokenizer = EsmTokenizer("vocab.txt", mask_token='#')
    return fs_tokenizer
