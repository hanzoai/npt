import os

def gen_job_name(config):
    name_to_merge = []

    str_train = str(config.model.augmentation_bert_mask_prob['train'])
    str_val = str(config.model.augmentation_bert_mask_prob['val'])
    str_type_embed = str(config.model.feature_type_embedding)[0]
    str_index_embed = str(config.model.feature_index_embedding)[0]
    
    name_to_merge.append(f'TORCH_SEED_{config.training.torch_seed}')
    name_to_merge.append(f'np_seed_{config.training.np_seed}')
    name_to_merge.append(f"{config.data.name.upper()}")
    name_to_merge.append(f'job_{os.environ.get("SLURM_JOBID","")}')
    name_to_merge.append(f'bs_{config.training.batch_size}')
    name_to_merge.append(f'lr_{config.training.lr}')
    name_to_merge.append(f'nsteps_{config.training.num_total_steps}')
    name_to_merge.append(f'hdim_{config.model.dim_hidden}')
    name_to_merge.append(f'trainmskprob_{str_train}')
    name_to_merge.append(f'valmskprob_{str_val}')
    name_to_merge.append(f'nheads_{config.model.num_heads}')
    name_to_merge.append(f'stcking_dpth_{config.model.stacking_depth}')
    name_to_merge.append(f'type_emb_{str_type_embed}')
    name_to_merge.append(f'index_emb_{str_index_embed}')

    return '__'.join(name_to_merge)