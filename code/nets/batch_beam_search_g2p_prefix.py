"""Parallel beam search module."""

import logging
from typing import Any, Dict, List, NamedTuple, Tuple, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from espnet.nets.e2e_asr_common import end_detect

from .beam_search_g2p import BeamSearchG2P, Hypothesis
from .scorers.prefix_score import PrefixScore

class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""

    yseq: torch.Tensor = torch.tensor([])  # (batch, maxlen)
    score: torch.Tensor = torch.tensor([])  # (batch,)
    length: torch.Tensor = torch.tensor([])  # (batch,)
    scores: Dict[str, torch.Tensor] = dict()  # values: (batch,)
    states: Dict[str, Dict] = dict()

    def __len__(self) -> int:
        """Return a batch size."""
        return len(self.length)


class BatchBeamSearchG2PPrefix(BeamSearchG2P):
    """Batch beam search implementation."""

    def batchfy(self, hyps: List[Hypothesis]) -> BatchHypothesis:
        """Convert list to batch."""
        if len(hyps) == 0:
            return BatchHypothesis()
        return BatchHypothesis(
            yseq=pad_sequence(
                [h.yseq for h in hyps], batch_first=True, padding_value=self.eos
            ),
            length=torch.tensor([len(h.yseq) for h in hyps], dtype=torch.int64),
            score=torch.tensor([h.score for h in hyps]),
            scores={k: torch.tensor([h.scores[k] for h in hyps]) for k in self.scorers},
            states={k: [h.states[k] for h in hyps] for k in self.scorers},
        )

    def _batch_select(self, hyps: BatchHypothesis, ids: List[int]) -> BatchHypothesis:
        return BatchHypothesis(
            yseq=hyps.yseq[ids],
            score=hyps.score[ids],
            length=hyps.length[ids],
            scores={k: v[ids] for k, v in hyps.scores.items()},
            states={
                k: [self.scorers[k].select_state(v, i) for i in ids]
                for k, v in hyps.states.items()
            },
        )

    def _select(self, hyps: BatchHypothesis, i: int) -> Hypothesis:
        return Hypothesis(
            yseq=hyps.yseq[i, : hyps.length[i]],
            score=hyps.score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            states={
                k: self.scorers[k].select_state(v, i) for k, v in hyps.states.items()
            },
        )

    def unbatchfy(self, batch_hyps: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        return [
            Hypothesis(
                yseq=batch_hyps.yseq[i][: batch_hyps.length[i]],
                score=batch_hyps.score[i],
                scores={k: batch_hyps.scores[k][i] for k in self.scorers},
                states={
                    k: v.select_state(batch_hyps.states[k], i)
                    for k, v in self.scorers.items()
                },
            )
            for i in range(len(batch_hyps.length))
        ]

    def batch_beam(
        self, weighted_scores: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-compute topk full token ids and partial token ids.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
                Its shape is `(n_beam, self.vocab_size)`.
            ids (torch.Tensor): The partial token ids to compute topk.
                Its shape is `(n_beam, self.pre_beam_size)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The topk full (prev_hyp, new_token) ids
                and partial (prev_hyp, new_token) ids.
                Their shapes are all `(self.beam_size,)`

        """
        top_ids = weighted_scores.view(-1).topk(self.beam_size)[1]
        # Because of the flatten above, `top_ids` is organized as:
        # [hyp1 * V + token1, hyp2 * V + token2, ..., hypK * V + tokenK],
        # where V is `self.n_vocab` and K is `self.beam_size`
        prev_hyp_ids = top_ids // self.n_vocab
        new_token_ids = top_ids % self.n_vocab
        return prev_hyp_ids, new_token_ids, prev_hyp_ids, new_token_ids

    def init_hyp(self, x: torch.Tensor, xp: torch.Tensor) -> BatchHypothesis:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature

        Returns:
            Hypothesis: The initial hypothesis.

        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            _x = xp if k == 'ctc' else x
            init_states[k] = d.batch_init_state(_x)
            init_scores[k] = 0.0
        return self.batchfy(
            [
                Hypothesis(
                    score=0.0,
                    scores=init_scores,
                    states=init_states,
                    yseq=torch.tensor([self.sos], device=x.device),
                )
            ]
        )

    def score_full(
        self, hyp: BatchHypothesis, x: torch.Tensor, text: Optional[str]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            if isinstance(d, PrefixScore):
                scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x, text)
            else:
                scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)
                # logging.info(f'scores[{k}] {scores[k].shape}')
        return scores, states

    def score_partial(
        self, hyp: BatchHypothesis, ids: torch.Tensor, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            ids (torch.Tensor): 2D tensor of new partial tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.batch_score_partial(
                hyp.yseq, ids, hyp.states[k], x
            )
        return scores, states

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        """Merge states for new hypothesis.

        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.

        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, v in part_states.items():
            new_states[k] = v
        return new_states
    
    @staticmethod
    def merge_scores(
        prev_scores: Dict[str, float],
        next_full_scores: Dict[str, torch.Tensor],
        full_idx: int,
        next_part_scores: Dict[str, torch.Tensor],
        part_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Merge scores for new hypothesis.

        Args:
            prev_scores (Dict[str, float]):
                The previous hypothesis scores by `self.scorers`
            next_full_scores (Dict[str, torch.Tensor]): scores by `self.full_scorers`
            full_idx (int): The next token id for `next_full_scores`
            next_part_scores (Dict[str, torch.Tensor]):
                scores of partial tokens by `self.part_scorers`
            part_idx (int): The new token id for `next_part_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are scalar tensors by the scorers.

        """
        new_scores = dict()
        for k, v in next_full_scores.items():
            if k == 'prefix_scorer':
                new_scores[k] = v[full_idx]
            else:
                new_scores[k] = prev_scores[k] + v[full_idx]
        for k, v in next_part_scores.items():
            new_scores[k] = prev_scores[k] + v[part_idx]
        return new_scores

    def search(self, running_hyps: BatchHypothesis, x: torch.Tensor, xp: torch.Tensor, text: str) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            BatchHypothesis: Best sorted hypotheses

        """

        n_batch = len(running_hyps)
        part_ids = None  # no pre-beam
        # batch scoring
        weighted_scores = torch.zeros(
            n_batch, self.n_vocab, dtype=x.dtype, device=x.device
        )
        scores, states = self.score_full(running_hyps, x.expand(n_batch, *x.shape), text)
        # logging.info(f'scores {scores} {self.full_scorers.keys()} {self.part_scorers.keys()} ')
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        # logging.info(f'weighted_scores {weighted_scores.shape} {weighted_scores} ')
        # partial scoring
        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]
        # NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
        # full-size score matrices, which has non-zero scores for part_ids and zeros
        # for others.
        # logging.info(f'{self.do_pre_beam} pre_beam_size {self.pre_beam_size} {self.pre_beam_score_key} scoring hyp {len(running_hyps)}')
        # logging.info(f' part_ids {part_ids} running_hyps {running_hyps} x {x.shape} {xp.shape}')

        part_scores, part_states = self.score_partial(running_hyps, part_ids, xp)
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]
        # add previous hyp scores
        weighted_scores += running_hyps.score.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)

        # TODO(karita): do not use list. use batch instead
        # see also https://github.com/espnet/espnet/pull/1402#discussion_r354561029
        # update hyps
        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)
        for (
            full_prev_hyp_id,
            full_new_token_id,
            part_prev_hyp_id,
            part_new_token_id,
        ) in zip(*self.batch_beam(weighted_scores, part_ids)):
            prev_hyp = prev_hyps[full_prev_hyp_id]
            best_hyps.append(
                Hypothesis(
                    score=weighted_scores[full_prev_hyp_id, full_new_token_id],
                    yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
                    scores=self.merge_scores(
                        prev_hyp.scores,
                        {k: v[full_prev_hyp_id] for k, v in scores.items()},
                        full_new_token_id,
                        {k: v[part_prev_hyp_id] for k, v in part_scores.items()},
                        part_new_token_id,
                    ),
                    states=self.merge_states(
                        {
                            k: self.full_scorers[k].select_state(v, full_prev_hyp_id)
                            for k, v in states.items()
                        },
                        {
                            k: self.part_scorers[k].select_state(
                                v, part_prev_hyp_id, part_new_token_id
                            )
                            for k, v in part_states.items()
                        },
                        part_new_token_id,
                    ),
                )
            )
        return self.batchfy(best_hyps)

    def forward(
        self, x: torch.Tensor, xp: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0, text: Optional[str] = None
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
                If maxlenratio<0.0, its absolute value is interpreted
                as a constant max output length.
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        # logging.info(f'x: {x.shape}')
        # set length bounds
        if maxlenratio == 0:
            maxlen = xp.shape[0]
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * xp.shape[0]))
        minlen = int(minlenratio * x.size(0))
        logging.info("decoder input length: " + str(x.shape[0]))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(x, xp)
        # logging.info(f'running_hyps {running_hyps}')

        ended_hyps = []
        for i in range(maxlen):
        # for i in range(5):
            logging.debug("position " + str(i))
            # logging.info(f'position {i}')
            best = self.search(running_hyps, x, xp, text)
            # post process of one iteration
            running_hyps = self.post_process(i, maxlen, maxlenratio, best, ended_hyps)
            # logging.info(f'running_hyps {text} {i} yseq {running_hyps.yseq}')
            # logging.info(f'running_hyps {text} {i} yseq {["".join([self.token_list[x] for x in y]) for y in running_hyps.yseq]}')
            # logging.info(f'running_hyps {text} {i} scores {running_hyps.scores}')
            # logging.info(f'running_hyps {i} length {running_hyps.length}')
            # logging.info(f'running_hyps {i} states {type(running_hyps.states["decoder"])} {len(running_hyps.states["decoder"])}')
            # logging.info(f'running_hyps {i} states[0] {type(running_hyps.states["decoder"][0])} {len(running_hyps.states["decoder"][0])}')
            # logging.info(f'running_hyps {i} states[0][0] {type(running_hyps.states["decoder"][0][0])} {running_hyps.states["decoder"][0][0].shape} {running_hyps.states["decoder"][0][1].shape}')
 
            # logging.info(f'running_hyps {i} ctc {running_hyps.ctc}')
            # if i >= 10:
            #     raise
            # end detection
            if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                logging.info(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logging.info("no hypothesis. Finish decoding.")
                break
            else:
                logging.debug(f"remained hypotheses: {len(running_hyps)}")

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, xp, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logging.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logging.info(f"total log probability: {best.score:.2f}")
        logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logging.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        return nbest_hyps

    def post_process(
        self,
        i: int,
        maxlen: int,
        maxlenratio: float,
        running_hyps: BatchHypothesis,
        ended_hyps: List[Hypothesis],
    ) -> BatchHypothesis:
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (BatchHypothesis): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.

        Returns:
            BatchHypothesis: The new running hypotheses.

        """
        n_batch = running_hyps.yseq.shape[0]
        logging.debug(f"the number of running hypothes: {n_batch}")
        if self.token_list is not None:
            logging.debug(
                "best hypo: "
                + "".join(
                    [
                        self.token_list[x]
                        for x in running_hyps.yseq[0, 1 : running_hyps.length[0]]
                    ]
                )
            )
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info("adding <eos> in the last position in the loop")
            yseq_eos = torch.cat(
                (
                    running_hyps.yseq,
                    torch.full(
                        (n_batch, 1),
                        self.eos,
                        device=running_hyps.yseq.device,
                        dtype=torch.int64,
                    ),
                ),
                1,
            )
            running_hyps.yseq.resize_as_(yseq_eos)
            running_hyps.yseq[:] = yseq_eos
            running_hyps.length[:] = yseq_eos.shape[1]

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a probmlem, number of hyps < beam)
        is_eos = (
            running_hyps.yseq[torch.arange(n_batch), running_hyps.length - 1]
            == self.eos
        )
        for b in torch.nonzero(is_eos, as_tuple=False).view(-1):
            hyp = self._select(running_hyps, b)
            ended_hyps.append(hyp)
        remained_ids = torch.nonzero(is_eos == 0, as_tuple=False).view(-1)
        return self._batch_select(running_hyps, remained_ids)
