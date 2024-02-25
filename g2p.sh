#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
expdir=exp           # Directory to save experiments.
asr_exp=
asr_tag=             # Suffix to the result dir for asr model training.
asr_stats_dir=       # Specify the directory path for ASR statistics.
data_feats=
asr_config=          # Config for asr model training.
ignore_init_mismatch=false      # Ignore initial mismatch
train_set=train
valid_set=valid
pretrained_unit_embed=

lexicon=
unit_name=
grapheme_name=
phoneme_name=
phoneme_token_type="char"
grapheme_token_type="char"
unit_token_type="char"
token_listdir=
acoustic_feature="unit"
python=python3
nlsyms_txt="none"
pretrained_model=

asr_text_fold_length=150
asr_speech_fold_length=51200

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


_asr_train_dir="${data_feats}/${train_set}"
_asr_valid_dir="${data_feats}/${valid_set}"
log "ASR Training: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"

_opts=
if [ -n "${asr_config}" ]; then
    _opts+="--config ${asr_config} "
fi

if [ ! -z ${pretrained_unit_embed} ]; then
    echo load from ${pretrained_unit_embed}
    _opts+="--pretrained_unit_embed ${pretrained_unit_embed} "
fi

if [ ! -z ${lexicon} ]; then
    echo load lexicon from ${lexicon}
    _opts+="--lexicon ${lexicon} "
fi

if [ -z ${asr_exp} ]; then
    asr_exp=${expdir}/${asr_tag}
fi

jobname="${asr_exp}/train.log"
phoneme_token_list="${token_listdir}"/${phoneme_name}_tokens.txt
grapheme_token_list="${token_listdir}"/${grapheme_name}_tokens.txt
unit_token_list="${token_listdir}"/${unit_name}_tokens.txt


export PYTHONPATH=$(pwd)
${python} -m espnet2.bin.launch \
    --cmd "${cuda_cmd} --name ${jobname}" \
    --log "${asr_exp}"/train.log \
    --ngpu "${ngpu}" \
    --num_nodes "${num_nodes}" \
    --init_file_prefix "${asr_exp}"/.dist_init_ \
    --multiprocessing_distributed true -- \
    PYTHONPATH=$(pwd) ${python} -m code.bin.g2p_train \
        --use_preprocessor true \
        --bpemodel "none" \
        --phoneme_name "phoneme" \
        --phoneme_token_type "${phoneme_token_type}" \
        --phoneme_token_list "${phoneme_token_list}" \
        --grapheme_name "grapheme" \
        --grapheme_token_type "${grapheme_token_type}" \
        --grapheme_token_list "${grapheme_token_list}" \
        --unit_name "unit" \
        --unit_token_type "${unit_token_type}" \
        --unit_token_list "${unit_token_list}" \
        --non_linguistic_symbols "${nlsyms_txt}" \
        --cleaner "none" \
        --g2p "none" \
        --train_data_path_and_name_and_type ${_asr_train_dir}/wav.scp,speech,sound \
        --train_data_path_and_name_and_type ${_asr_train_dir}/${unit_name},unit,text \
        --train_data_path_and_name_and_type ${_asr_train_dir}/${grapheme_name},grapheme,text \
        --train_data_path_and_name_and_type ${_asr_train_dir}/${phoneme_name},phoneme,text \
        --train_shape_file ${asr_stats_dir}/train/speech_shape \
        --train_shape_file ${asr_stats_dir}/train/${unit_name}_shape \
        --train_shape_file ${asr_stats_dir}/train/${grapheme_name}_shape \
        --train_shape_file ${asr_stats_dir}/train/${phoneme_name}_shape \
        --allow_variable_data_keys true \
        --valid_data_path_and_name_and_type "${_asr_valid_dir}/wav.scp,speech,sound" \
        --valid_data_path_and_name_and_type "${_asr_valid_dir}/${unit_name},unit,text" \
        --valid_data_path_and_name_and_type "${_asr_valid_dir}/${grapheme_name},grapheme,text" \
        --valid_data_path_and_name_and_type "${_asr_valid_dir}/${phoneme_name},phoneme,text" \
        --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
        --valid_shape_file "${asr_stats_dir}/valid/${unit_name}_shape" \
        --valid_shape_file "${asr_stats_dir}/valid/${grapheme_name}_shape" \
        --valid_shape_file "${asr_stats_dir}/valid/${phoneme_name}_shape" \
        --resume true \
        --init_param ${pretrained_model} \
        --ignore_init_mismatch ${ignore_init_mismatch} \
        --fold_length "${asr_speech_fold_length}" \
        --fold_length "${asr_text_fold_length}" \
        --fold_length "${asr_text_fold_length}" \
        --output_dir "${asr_exp}" \
        --acoustic_feature "${acoustic_feature}" \
        ${_opts}

