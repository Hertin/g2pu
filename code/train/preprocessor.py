import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Union, Optional

import numpy as np
import scipy.signal
import soundfile
from typeguard import check_argument_types, check_return_type

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet2.text.token_id_converter import TokenIDConverter

from transformers import AutoTokenizer
import fairseq
from collections import defaultdict
from tqdm import tqdm

class AbsPreprocessor(ABC):
    def __init__(self, train: bool):
        self.train = train

    @abstractmethod
    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError

class G2PPreprocessor(AbsPreprocessor):
    def __init__(
        self,
        train: bool,
        phoneme_token_type: str = None,
        phoneme_token_list: Union[Path, str, Iterable[str]] = None,
        grapheme_token_type: str = None,
        grapheme_token_list: Union[Path, str, Iterable[str]] = None,
        unit_token_type: str = None,
        unit_token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        speech_volume_normalize: float = None,
        phoneme_name: str = "phoneme",
        grapheme_name: str = "grapheme",
        unit_name: str = "unit",
        acoustic_feature: str = "unit",
        roberta: Optional[str] = None,
        data: Optional[str] = None,
        bert_model: Optional[str] = None,
        lexicon: Optional[str] = None,
        st: bool = False,
        st_prob: Optional[float] = None
    ):
        super().__init__(train)
        self.train = train
        self.phoneme_name = phoneme_name
        self.grapheme_name = grapheme_name
        self.unit_name = unit_name
        self.acoustic_feature = acoustic_feature
        self.phoneme_cleaner = TextCleaner(text_cleaner)
        self.st = st
        self.st_prob = st_prob
        if self.st:
            logging.info(f'use self training transcripts with probability {self.st_prob}')
        else:
            logging.info('do not use self training transcripts')

        self.phoneme_tokenizer = build_tokenizer(
            token_type=phoneme_token_type,
            bpemodel=bpemodel,
            delimiter=delimiter,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            g2p_type=g2p_type,
        )
        self.phoneme_token_id_converter = TokenIDConverter(
            token_list=phoneme_token_list,
            unk_symbol=unk_symbol,
        )

        self.grapheme_cleaner = TextCleaner(text_cleaner)
        if bert_model is None: # just for compatibility
            bert_model = 'bert-base-chinese'
        self.bert_model = bert_model
        self.grapheme_tokenizer = AutoTokenizer.from_pretrained(bert_model)
        # self.grapheme_tokenizer = build_tokenizer(
        #     token_type=grapheme_token_type,
        #     bpemodel=bpemodel,
        #     delimiter=delimiter,
        #     space_symbol=space_symbol,
        #     non_linguistic_symbols=non_linguistic_symbols,
        #     g2p_type=g2p_type,
        # )
        # self.grapheme_token_id_converter = TokenIDConverter(
        #     token_list=grapheme_token_list,
        #     unk_symbol=unk_symbol,
        # )

        if acoustic_feature == 'unit':
            self.unit_cleaner = TextCleaner(text_cleaner)
            self.unit_tokenizer = build_tokenizer(
                token_type=unit_token_type,
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
            )
            self.unit_token_id_converter = TokenIDConverter(
                token_list=unit_token_list,
                unk_symbol=unk_symbol,
            )
        elif acoustic_feature == 'unit_roberta':
            assert roberta is not None
            assert data is not None
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [roberta], arg_overrides={'data': data})
            del model, cfg
            self.task = task
    
        self.lexicon = None
        if lexicon is not None:
            logging.info(f'load lexicon from {lexicon}')
            _lexicon = defaultdict(set)
            self.lexicon_path = lexicon
            with open(lexicon, 'r') as f:
                for l in tqdm(f):
                    g, p = l.strip().split('\t')
                    _lexicon[g].add(p)
            self.lexicon = {}
            for k, v in _lexicon.items():
                self.lexicon[k] = list(v)


    def _unit_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.acoustic_feature == 'unit':
            if self.unit_name in data and self.unit_tokenizer is not None:
                text = data[self.unit_name]
                text = self.unit_cleaner(text)
                tokens = text.split()
                text_ints = self.unit_token_id_converter.tokens2ids(tokens)
                data[self.unit_name] = np.array(text_ints, dtype=np.int64)
        elif self.acoustic_feature == 'unit_roberta':
            if self.unit_name in data:
                text = data[self.unit_name]
                # logging.info(f'unit_roberta {text}')
                text_ints = self.task.dictionary.encode_line(
                    f'{self.task.dictionary.bos_word} {text} {self.task.dictionary.eos_word}', 
                    add_if_not_exist=False, append_eos=False).numpy()
                data[self.unit_name] = np.array(text_ints, dtype=np.int64)
        
        # logging.info(f'unit={text} || {tokens} || {text_ints} use st trans')
        return data

    def _phoneme_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.phoneme_name in data and self.phoneme_tokenizer is not None:
            r = random.random()
            if self.lexicon is None or (self.st and r < self.st_prob):
                text = data[self.phoneme_name].replace(' ', '') 
                # logging.info(f'r={r} use st trans')
            else:
                graphemes = data[self.grapheme_name].split('|')
                phonemes = []
                for i, phrase in enumerate(graphemes):
                    # logging.info(f'phrase: {self.lexicon[phrase]}')
                    p = random.choice(self.lexicon[phrase])
                    phonemes.append(p)
                text = ' '.join(''.join(phonemes))

            text = self.phoneme_cleaner(text)
            tokens = self.phoneme_tokenizer.text2tokens(text)
            # logging.info(f'phoneme={text} || {tokens} use st trans')

            text_ints = self.phoneme_token_id_converter.tokens2ids(tokens)
            # logging.info(f'phoneme={text} || {tokens} || {text_ints} use st trans')
            data[self.phoneme_name] = np.array(text_ints, dtype=np.int64)
        return data

    def _grapheme_process(
        self, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if self.grapheme_name in data and self.grapheme_tokenizer is not None:
            text = data[self.grapheme_name].replace('|', '')
            text = self.grapheme_cleaner(text)
            # tokens = self.grapheme_tokenizer.text2tokens(text)
            # text_ints = self.grapheme_token_id_converter.tokens2ids(tokens)
            # data[self.grapheme_name] = np.array(text_ints, dtype=np.int64)
            text_ints = self.grapheme_tokenizer.encode(text, return_tensors='np').squeeze()
            # logging.info(f'grapheme {text} || {text_ints}')
            data[self.grapheme_name] = text_ints
        return data

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()
        if self.acoustic_feature in ['unit', 'unit_roberta']:
            data = self._unit_process(data)
            del data['speech']
        if self.acoustic_feature in ['speech', 'speech_wcnn', 'speech_cnn', 'cnn']:
            del data['unit']
        # logging.info(f'prev data {data}')
        # The order of self._phoneme_process and self._grapheme_process matters
        data = self._phoneme_process(data)
        # logging.info(f'mid data {data}')

        data = self._grapheme_process(data)
        # logging.info(f'post data {data}')

        # raise 
        assert check_return_type(data)
        return data

