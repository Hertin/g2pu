#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=0
stop_stage=100
inference_nj=1
inference_config=
inference_tag=
inference_asr_model=valid.acc.ave.pth
batch_size=1
test_sets=
beam_size=10
inference_script=g2p_inference
prefix_weight=0
dictionary=
u_checkpoint=
s_checkpoint=
g_checkpoint=

expdir=exp           # Directory to save experiments.
asr_exp=
asr_tag=             # Suffix to the result dir for asr model training.
# asr_stats_dir=       # Specify the directory path for ASR statistics.
data_feats=
# ignore_init_mismatch=false      # Ignore initial mismatch
# train_set=train
# valid_set=valid
pretrained_unit_embed=

# lexicon=
unit_name=
grapheme_name=
phoneme_name=
phoneme_token_type="char"
grapheme_token_type="char"
unit_token_type="char"
token_listdir=
# acoustic_feature="unit"
python=python3
nlsyms_txt="none"
pretrained_model=
score_opts=
cleaner="none"
ref_token_type=word

# asr_text_fold_length=150
# asr_speech_fold_length=51200

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# _asr_train_dir="${data_feats}/${train_set}"
# _asr_valid_dir="${data_feats}/${valid_set}"
# log "ASR Training: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"

_opts=

# if [ ! -z ${pretrained_unit_embed} ]; then
#     echo load from ${pretrained_unit_embed}
#     _opts+="--pretrained_unit_embed ${pretrained_unit_embed} "
# fi

# if [ ! -z ${lexicon} ]; then
#     echo load lexicon from ${lexicon}
#     _opts+="--lexicon ${lexicon} "
# fi

if [ -z ${asr_exp} ]; then
    asr_exp=${expdir}/${asr_tag}
fi

# jobname="${asr_exp}/train.log"
# phoneme_token_list="${token_listdir}"/${phoneme_name}_tokens.txt
# grapheme_token_list="${token_listdir}"/${grapheme_name}_tokens.txt
# unit_token_list="${token_listdir}"/${unit_name}_tokens.txt





_cmd="${decode_cmd}"
_ngpu=0

_opts=
if [ -n "${inference_config}" ]; then
    _opts+="--config ${inference_config} "
fi

if [ ! -z ${pretrained_unit_embed} ]; then
    echo load from ${pretrained_unit_embed}
    _opts+="--pretrained_unit_embed ${pretrained_unit_embed} "
fi

if [ ! -z ${u_checkpoint} ]; then
    echo load unit encoder from ${u_checkpoint}
    _opts+="--u_checkpoint ${u_checkpoint} "
fi

if [ ! -z ${s_checkpoint} ]; then
    echo load speech encoder from ${s_checkpoint}
    _opts+="--s_checkpoint ${s_checkpoint} "
fi

if [ ! -z ${g_checkpoint} ]; then
    echo load grapheme encoder from ${g_checkpoint}
    _opts+="--g_checkpoint ${g_checkpoint} "
fi

if [ ! -z ${beam_size} ]; then
    echo "set beam_size to ${beam_size}"
    _opts+="--beam_size ${beam_size} "
fi

if [ ! -z ${dictionary} ]; then
    echo "use dictioanry ${dictionary}"
    _opts+="--dictionary ${dictionary} "
fi

if [ ! -z ${prefix_weight} ]; then
    echo "set prefix_weight ${prefix_weight}"
    _opts+="--prefix_weight ${prefix_weight} "
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for dset in ${test_sets}; do
        log "Stage 1: Decoding: training_dir=${asr_exp}"
        _data="${data_feats}/${dset}"
        _dir="${asr_exp}/${inference_tag}/${dset}"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        _scp=wav.scp
        _type=sound

        # 1. Split the key file
        key_file=${_data}/${_scp}
        split_scps=""

        _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")

        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
        rm -f "${_logdir}/*.log"
        # shellcheck disable=SC2046,SC2086
        export PYTHONPATH=$(pwd)
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
            ${python} -m code.bin.${inference_script} \
                --batch_size ${batch_size} \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                --data_path_and_name_and_type "${_data}/${unit_name},unit,text" \
                --data_path_and_name_and_type "${_data}/${grapheme_name},grapheme,text" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --g2p_train_config "${asr_exp}"/config.yaml \
                --g2p_model_file "${asr_exp}"/"${inference_asr_model}" \
                --output_dir "${_logdir}"/output.JOB \
                ${_opts} || { cat $(grep -l -i error "${_logdir}"/asr_inference.*.log) ; exit 1; }


        # 4. Concatenates the output files from each jobs
        for f in token token_int score text; do
            if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
              for i in $(seq "${_nj}"); do
                  cat "${_logdir}/output.${i}/1best_recog/${f}"
              done | sort -k1 >"${_dir}/${f}"
            fi
        done
        rm -rf ${_logdir}
    done

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Scoring"
    for dset in ${test_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${asr_exp}/${inference_tag}/${dset}"

        for _type in cer wer; do

            _scoredir="${_dir}/score_${_type}"
            mkdir -p "${_scoredir}"

            if [ "${_type}" = wer ]; then
                # Tokenize text to word level
                tmp_file=$(mktemp)
                cat ${_data}/${phoneme_name} | sort -k1 > $tmp_file
                paste \
                    <(<${tmp_file} \
                          ${python} -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type ${ref_token_type} \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              --cleaner "${cleaner}" \
                              ) \
                    <(<"${_data}/utt2spk" sort -k1 | awk '{ print "(" $2 "-" $1 ")" }') \
                         >"${_scoredir}/ref.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/text"  \
                          ${python} -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type char \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              ) \
                    <(<"${_data}/utt2spk" sort -k1 | awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"
                rm $tmp_file

            elif [ "${_type}" = cer ]; then
                # Tokenize text to char level
                tmp_file=$(mktemp)
                cat ${_data}/${phoneme_name} | sort -k1 > $tmp_file
                paste \
                    <(<"$tmp_file"  \
                          ${python} -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type ${ref_token_type} \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              --cleaner "${cleaner}" \
                              ) \
                    <(<"${_data}/utt2spk" sort -k1 | awk '{ print "(" $2 "-" $1 ")" }') \
                       >"${_scoredir}/ref.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/text"  \
                          ${python} -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type char \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true \
                              ) \
                    <(<"${_data}/utt2spk" sort -k1 | awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"
            rm $tmp_file
            fi

            sclite \
                ${score_opts} \
                -r "${_scoredir}/ref.trn" trn \
                -h "${_scoredir}/hyp.trn" trn \
                -i rm -o all stdout > "${_scoredir}/result.txt"

            log "Write ${_type} result in ${_scoredir}/result.txt"
            grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
        done
    done

    # Show results in Markdown syntax
    utils/show_asr_result.sh "${asr_exp}" > "${asr_exp}"/RESULTS.md
    cat "${asr_exp}"/RESULTS.md

fi

