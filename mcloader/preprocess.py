import os
import re
import regex
import json
from tqdm import tqdm
from typing import Dict, List, Union

import torch
from clip.simple_tokenizer import SimpleTokenizer

from .prompt_template import prompt_templates

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as rf:
        data = json.load(rf)
    return data

def megasplit(pattern, string):
    splits = list((m.start(), m.end()) for m in re.finditer(pattern, string))
    starts = [0] + [i[1] for i in splits]
    ends = [i[0] for i in splits] + [len(string)]
    return [string[start:end] for start, end in zip(starts, ends)]


class SentPreProcessor(object):

    def __init__(self, root, dataset: str="Kinetics"):
        assert dataset in ['UCF101', 'HMDB51', 'Kinetics', "Kinetics100_base",
                           'Kinetics100_test', 'k100_support_query', 'kinetics400_openset']
        self.root = root

        with open(os.path.join(self.root, "labels.txt"), "r") as rf:
            data = rf.readlines()
        _lines = [l.split() for l in data]
        self.categories = []
        for id, l in enumerate(_lines):
            name = '_'.join(l)
            name = name.replace("_", ' ')
            self.categories.append({"id": id, "name": name})
        self.categories.sort(key=lambda x: x["id"])

        self.drop_keys = ['External links', 'References', 'Further reading', 'Bibliography']
        self._tokenizer = SimpleTokenizer()
        self.SEP_TOKENS = [267, 269]  # [',', '.']
        self.wikis = None

    def get_naive_text(self):
        naive_text = [self.gen_naive_desc(id) for id in range(len(self.categories))]
        return naive_text

    def get_clip_text(self):
        if self.wikis is None:
            self.wikis = [self._parse_wiki(id) for id in tqdm(range(len(self.categories)))]
        naive_text = [self.gen_naive_desc(id) for id in range(len(self.categories))]
        wiki_text = [self._get_text(wiki) for wiki in self.wikis]
        return [naive_text[i] + wiki_text[i] for i in range(len(self.categories))]

    def split_sent(self, texts: List[str]) -> List[List[str]]:
        pat = re.compile(r'(?<!\w\.\w.)(?<!([A-Z][a-z])|([A-Z])\.)(?<=\.|\?)(?=[\sA-Z])', re.X)
        sents = []
        for text in texts:
            try:
                split_text = pat.split(text)
            except:
                split_text = regex.split(r'(?<!\w\.\w.)(?<!([A-Z][a-z])|([A-Z])\.)(?<=\.|\?)(?=[\sA-Z])', text,
                                         flags=regex.VERSION1)
            split_text = [s.strip() for s in split_text if s is not None and s.strip() != '']
            sents.append(split_text)
        return sents

    def gen_naive_desc(self, id):
        texts = [template.format(self.categories[id]['name'] + ' ') for template in prompt_templates]
        return '\n'.join(texts)

    def tokenize(self, texts: Union[str, List[str], List[List[str]]], context_length=75):
        """
        modified from CLIP

        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        def _tokenize(texts):
            sot_token = self._tokenizer.encoder["<|startoftext|>"]  # 49406
            eot_token = self._tokenizer.encoder["<|endoftext|>"]  # 49407
            all_tokens = [[sot_token] + self._tokenizer.encode(text)[:context_length] + [eot_token] for text in
                            texts]
            result = torch.zeros(len(all_tokens), context_length + 2, dtype=torch.long)
            for i, tokens in enumerate(all_tokens):
                if len(tokens) > context_length + 2:
                    raise RuntimeError(
                        f"Input {texts[i]} is too long for context length {context_length}")
                result[i, :len(tokens)] = torch.tensor(tokens)
            return result
        
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts[0], List):
            return [_tokenize(text) for text in tqdm(texts)]
        return _tokenize(texts)

    def _get_text(self, wiki: Dict):
        # use all key part of each wiki text except those in drop_keys
        text = wiki["summary"] + "\n"
        text += "\n".join([v for k, v in wiki.items() if k not in ["summary"] + self.drop_keys])
        return text

    def _parse_wiki(self, id) -> Dict:
        try:
            with open(os.path.join(self.root, "wiki", f"desc_{id}.txt")) as rf:
                lines = rf.readlines()
        except UnicodeDecodeError:
            with open(os.path.join(self.root, "wiki", f"desc_{id}.txt"), encoding='gbk') as rf:
                lines = rf.readlines()
        lines = [d.strip() for d in lines if d.strip() != '']
        ret_dict = {}
        key = "summary"
        val = ""
        for line in lines:
            if line[:2] == "==":
                ret_dict[key] = val.strip()
                key = line.strip('= ')
                val = ""
            else:
                val += line + '\n'
        ret_dict[key] = val.strip()
        return ret_dict
