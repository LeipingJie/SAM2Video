#!/bin/bash

types=("tiny" "small" "large")
datasets=("vmd" "visha")
# datasets=("visha")
# prompts=("point" "mask")
prompts=("point")
npts=(5 10 20 30 40 50)
# npts=(20 30 40 50)
# npts=(5 10)

for type in "${types[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for prompt in "${prompts[@]}"
            do
                for npt in "${npts[@]}"
                    do
                        echo "**** cmd : $type $dataset $prompt"
                        echo "python video_test.py --type $type --prompt $prompt --dataset $dataset --npts $npt"
                        python video_test.py --type $type --prompt $prompt --dataset $dataset --npts $npt
                        # wait
                        # echo "cmd : $type $dataset $prompt"
                    done
            done
        done
    done
done