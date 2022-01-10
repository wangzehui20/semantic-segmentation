import yaml
import os
import os.path as osp


def _update_dict(d: dict, params: dict):
    return d


def save_config(config, directory, name='config.yaml'):
    os.makedirs(directory)
    fp = osp.join(directory, name)

    with open(fp, 'w') as f:
        yaml.dump(config, f)
    

def parse_config(**kwargs):
    # get config path
    cfg_path = kwargs['config']

    # read config
    with open(cfg_path) as cfg:
        cfg_yaml = yaml.load(cfg, Loader=yaml.FullLoader)

    # override passed parameters in config
    update_cfg = _update_dict(cfg_yaml, kwargs)
    
    return update_cfg