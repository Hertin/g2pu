import torch
import logging
import copy
from typing import List, Optional, Tuple, Any
from espnet.nets.scorer_interface import BatchScorerInterface




class PrefixScore(BatchScorerInterface):
    """Batch partial scorer interface for beam search."""
    def __init__(self, dictionary, converter, max_phrase_size=10, separator=''):
        self.dictionary = dictionary
        self.converter = copy.deepcopy(converter)
        self.max_phrase_size =  max_phrase_size

        self.blank_token_id = 0
        self.unk_token_id = 1
        self.sos_token_id = len(self.converter.token_list) - 1
        # hack the token list to use only single characters
        self.converter.token_list[self.blank_token_id] = chr(0)
        self.converter.token_list[self.unk_token_id] = chr(1)
        self.converter.token_list[self.sos_token_id] = chr(2)

        self.sos_token = self.converter.token_list[self.sos_token_id]
        self.eos_token = self.sos_token
        assert self.sos_token not in self.dictionary
        self.dictionary[self.sos_token] = self.sos_token
        self.separator = separator
        logging.info(f'separator is : "{self.separator}"')


    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        text: List[str]
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            next_tokens (torch.Tensor): torch.int64 tokens to score (n_batch, n_token).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for ys that has a shape `(n_batch, n_vocab)`
                and next states for ys
        """
        text = self.sos_token + text + self.eos_token
        text = text.upper().replace(' ', '')
        hyps = []
        scores = []

        for y in ys:
            prev_hyp = self.separator.join(self.converter.ids2tokens(y))

            # hyps.append(hyp_cleaned)
            _scores = []
            for i, next_token in enumerate(self.converter.token_list):
                hyp = self.separator.join([prev_hyp, next_token])
                # logging.info(f'next_token {next_token} prev_hyp: {prev_hyp} hyp: {hyp}')

                ispf, _, __ = self.is_prefix(text, hyp)
                
                # logging.info(f'next_token {next_token} txt: "{text}" prev_hyp: {prev_hyp} hyp: "{hyp}" ispf "{ispf}"')
                score = 0. if ispf else -10.
                # if ispf or hyp.endswith(chr(2)):
                # if hyp == '\x02マルノ\x02':
                    # logging.info(f'text{text}: next_token {next_token} prev_hyp: {prev_hyp} hyp: {hyp} score: {score}')
                    # for j in range(5,len(hyp)+1):
                    #     h = hyp[:j]
                    #     ispf, _, __ = self.is_prefix(text, h)
                    #     logging.info(f'text {list(text)} | {j} | hyp: {list(h)} {ispf} {_} {__} ')
                    # raise
                _scores.append(score)
            scores.append(_scores)
        scores = torch.FloatTensor(scores).to(xs)

        return scores, None


    def isprefix(self, text, prefix, cache):
        if len(prefix) == 0:
            return True, [], []
        if len(text) == 0: # len(prefix) != 0
            return False, None, None
        for i in range(0, min(self.max_phrase_size, len(text)+1)):
            phrase, remain = text[:i], text[i:]
            if phrase not in self.dictionary:
                continue
            for pf in self.dictionary[phrase]:
                if pf.startswith(prefix):
                    return True, [phrase], [prefix]
                if not prefix.startswith(pf):
                    continue
                prefix_remain = prefix.replace(pf, '', 1)
                key = (remain, prefix_remain)
                if key in cache:
                    ispf, text_segment, phoneme_segment = cache[key]
                else:
                    ispf, text_segment, phoneme_segment = self.isprefix(remain, prefix_remain, cache)
                    cache[key] = ispf, text_segment, phoneme_segment
                if not ispf:
                    continue
                return True, [phrase] + text_segment, [pf] + phoneme_segment
        return False, None, None

    def is_prefix(self, text, prefix):
        CACHE = {}
        return self.isprefix(text, prefix, CACHE)
