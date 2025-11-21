import os
import yaml
import argparse
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--mmaigc_dataset_yml",
        type=str,
        default=None,
        help="the yaml config file for mmagic 's clip data real degradation",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--deg_file_path",
        type=str,
        default=None,
        required=True,
        help="The path of the deg yaml."
    )
    parser.add_argument(
        "--dataset_txt_paths",
        type=str,
        default=None,
        required=True,
        help="The path of the images."
    )
    parser.add_argument('--highquality_dataset_txt_paths', 
                        type=str, 
                        nargs='?', 
                        default=None, 
                        help='Paths to high quality dataset txt files'
    )
    parser.add_argument(
        "--null_text_ratio",
        type=float,
        default=0,
        help="null_text_ratio",
    )
    parser.add_argument(
        "--use_qwen",
        default=False,
        action="store_true",
        help="Whether to use qwen to get prompt",
    )
    # gradient_accumulation_steps
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient_accumulation_steps",
    )
    parser.add_argument(
        "--offload_dis_t5",
        default=False,
        action="store_true",
        help="Whether to offload dis's t5 to save gpu memory.",
    )
    args = parser.parse_args()
    return args

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])