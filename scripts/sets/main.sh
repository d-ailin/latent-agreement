models=(
    clip_vit-l-14
)

models_str=""
model_i=0
echo $(( ${#models[@]} - 1))
for m in ${models[@]}; do
    if [[ $model_i -eq $(( ${#models[@]} - 1)) ]];then
        models_str+="[${m}]"
    else
        models_str+="[${m}],"
    fi
    model_i=$(( $model_i + 1 ))
    echo $model_i
done

echo $models_str


exp_tag=main_single
tag=main_single
cnn_arg_models_str='+run.set.exp_tag='${exp_tag}' run.pretrain.models='${models_str}' run.algo.score=ndcg_rank run/pretrain=single run.set.neighbor_pool_size=10000 run.set.eval_ood=False'
vit_arg_models_str='+run.set.exp_tag='${exp_tag}' run.pretrain.models='${models_str}' run.algo.score=ndcg_rank run/pretrain=single run.set.neighbor_pool_size=10000 run.set.eval_ood=False'

