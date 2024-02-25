#!/usr/bin/env bash
PYTHON_VIRTUAL_ENVIRONMENT=espnet-g2p
CONDA_ROOT=/nobackup/users/heting/espnet/tools/conda

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

set -e
set -u
set -o pipefail

ulimit -s unlimited

test=
ctc_weight=0
inference_nj=80
ngpu=1
. utils/parse_options.sh

export MAIN_ROOT=/nobackup/users/heting/espnet-v.202209

me=$(basename "$0")
name=${me%.*}

train_set=train
valid_set=valid
test_sets=valid

unit_name=unit
phoneme_name=phg2pw
grapheme_name=hanzi
acoustic_feature=unit
asr_config=conf/${name}.yaml
asr_tag=$(basename ${asr_config%.*})
data_feats=dump/aishellX

if [ -z $test ]; then
    asr_stats_dir=dump/aishellX/shape
    token_listdir=dump/aishellX/token_list
    ./g2p.sh \
        --ngpu ${ngpu} \
        --asr_config "${asr_config}"                       \
        --data_feats "${data_feats}"                       \
        --asr_stats_dir "${asr_stats_dir}"                 \
        --train_set "${train_set}"                         \
        --valid_set "${valid_set}"                         \
        --token_listdir "${token_listdir}"                 \
        --asr_tag "${asr_tag}" \
        --acoustic_feature ${acoustic_feature} \
        --unit_name "${unit_name}" \
        --phoneme_name "${phoneme_name}" \
        --grapheme_name "${grapheme_name}"
else
    suf=
    _opts=
    u_checkpoint=exp/$name/valid.cer_ctc.best.pth
    infer_phoneme_name=pinyin
    beam_size=10
    ctc_weights=(${ctc_weight})
    inference_script=g2p_inference_prefix
    dictionary=dump/aishellX/dict/mandarin.dict
    prefix_weight=0.5

    if [ ! -z $u_checkpoint ]; then
        _opts="${_opts} --u_checkpoint ${u_checkpoint}"
    fi
    
    for ctc_weight in ${ctc_weights[@]}; do
        inference_config=$(mktemp)
        cat <(grep -v ctc conf/decode.yaml) <(echo "ctc_weight: ${ctc_weight}") > $inference_config
        
        echo decoding using $inference_config "$(cat $inference_config)"
        inference_tag=${unit_name}_${phoneme_name}_${infer_phoneme_name}_ctcw${ctc_weight}_prefix${prefix_weight}_bsz${beam_size}${suf}
        ./dec.sh \
            --stage 0 --stop-stage 2 \
            --inference_nj ${inference_nj} \
            --inference_config "${inference_config}" \
            --inference_tag ${inference_tag} \
            --test_sets "${test_sets}" \
            --data_feats "${data_feats}" \
            --asr_tag "${asr_tag}" \
            --grapheme_name "${grapheme_name}" \
            --phoneme_name "${infer_phoneme_name}" \
            --unit_name "${unit_name}" \
            --inference_asr_model valid.cer.best.pth \
            --beam_size ${beam_size} \
            --inference_script ${inference_script} \
            --dictionary ${dictionary} \
            --prefix_weight ${prefix_weight} \
            ${_opts}

        rm $inference_config
    done
fi
