import pickle

from bpref_v2.reward_learning.algos import (
    CLIPLearner,
    DiscriminatorLearner,
    DrSLearner,
    R2RRankLearner,
    REDSCNNLearner,
    REDSLearner,
    REDSNOEPICLearner,
    REDSNOSUPCONLearner,
    REDSNoTransLearner,
)

observation_dim = (224, 224, 3)
action_dim = 8

###############################################################################################################
######################################## Base function ########################################################
###############################################################################################################


def load_reward_model(rm_type, task_name, ckpt_path=None, args=None, obs_dim=None, act_dim=None):
    if ckpt_path is not None and not ckpt_path.is_dir():
        with ckpt_path.open("rb") as fin:
            checkpoint_data = pickle.load(fin)
            config, state = checkpoint_data["config"], checkpoint_data["state"]
    elif args is not None:
        config = dict(args)
        state = None

    if rm_type == "REDS":
        import transformers

        cfg = transformers.GPT2Config(**config)
        reward_model = REDSLearner(cfg, task_name, observation_dim, action_dim, state=state)

    if rm_type == "REDSCNN":
        import transformers

        cfg = transformers.GPT2Config(**config)
        reward_model = REDSCNNLearner(cfg, task_name, observation_dim, action_dim, state=state)
    if rm_type == "REDSNOEPIC":
        import transformers

        cfg = transformers.GPT2Config(**config)
        reward_model = REDSNOEPICLearner(cfg, task_name, observation_dim, action_dim, state=state)
    if rm_type == "REDSNOSUPCON":
        import transformers

        cfg = transformers.GPT2Config(**config)
        reward_model = REDSNOSUPCONLearner(cfg, task_name, observation_dim, action_dim, state=state)
    if rm_type == "REDSNoTrans":
        import transformers

        cfg = transformers.GPT2Config(**config)
        reward_model = REDSNoTransLearner(cfg, task_name, observation_dim, action_dim, state=state)

    elif rm_type == "DISC":
        cfg = DiscriminatorLearner.get_default_config()
        cfg.update(config)
        reward_model = DiscriminatorLearner(
            cfg, task_name, obs_dim or observation_dim, act_dim or action_dim, state=state
        )

    elif rm_type == "DRS":
        cfg = DrSLearner.get_default_config().unlock()
        cfg.update(config)
        reward_model = DrSLearner(cfg, task_name, obs_dim or observation_dim, act_dim or action_dim, state=state)

    elif rm_type == "R2R":
        cfg = R2RRankLearner.get_default_config().unlock()
        cfg.update(config)
        reward_model = R2RRankLearner(cfg, task_name, obs_dim or observation_dim, act_dim or action_dim, state=state)

    elif rm_type == "CLIP":
        cfg = CLIPLearner.get_default_config()
        cfg.transfer_type = "clip_vit_b16"
        reward_model = CLIPLearner(cfg)

    elif rm_type == "LIV":
        cfg = CLIPLearner.get_default_config()
        cfg.transfer_type = "liv"
        reward_model = CLIPLearner(cfg)

    elif rm_type == "none":
        reward_model = None

    return reward_model
