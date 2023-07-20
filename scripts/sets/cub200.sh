
seeds=(0 1 2 3 4)
for s in ${seeds[@]}; do
    echo $s

    python main.py -m run=cub200\
    run.set.neighbor_source=train\
    run.base_model.model_seed=501,502\
    run.set.split_seed=$s\
    +run.set.other=splitseed:${s}\
    ${vit_arg_models_str}\
    run.set.neighbor_pool_size=5994\  # force to use 5994, less than 10000


    python main.py -m run=cub200\
        run/base_model=cub200_cnn\
        run.set.neighbor_source=train\
        run.base_model.model_seed=503,504,505\
        run.set.split_seed=$s\
        +run.set.other=splitseed:${s}\
        ${cnn_arg_models_str}\
        run.set.neighbor_pool_size=5994\


done
