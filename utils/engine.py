import os
import time
import argparse
import yaml
import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from .logger import get_logger
from collections import OrderedDict
import datetime
import random

logger = get_logger()


def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if model_file is None:
        return model

    if isinstance(model_file, str):
        state_dict = torch.load(model_file, map_location="cpu")
        if "model" in state_dict.keys():
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict.keys():
            state_dict = state_dict["module"]
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = "module." + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=True)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend
        )
    )

    return model


def parse_devices(input_devices):
    if input_devices.endswith("*"):
        devices = list(range(torch.cuda.device_count()))
        return devices
    devices = []
    for d in input_devices.split(","):
        if "-" in d:
            start_device, end_device = d.split("-")[0], d.split("-")[1]
            assert start_device != ""
            assert end_device != ""
            start_device, end_device = int(start_device), int(end_device)
            assert start_device < end_device
            assert end_device < torch.cuda.device_count()
            for sd in range(start_device, end_device + 1):
                devices.append(sd)
        else:
            device = int(d)
            # assert device < torch.cuda.device_count()
            devices.append(device)

    logger.info("using devices {}".format(", ".join([str(d) for d in devices])))

    return devices


def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.system("rm -rf {}".format(target))
    os.system("ln -s {} {}".format(src, target))


def ensure_dir(path):
    if not os.path.isdir(path):
        try:
            sleeptime = random.randint(0, 3)
            time.sleep(sleeptime)
            os.makedirs(path)
        except:
            print("conflict !!!")


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)

    return tensor

class State(object):
    def __init__(self):
        self.epoch = 1
        self.iteration = 0
        self.dataloader = None
        self.model = None
        self.optimizer = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            assert k in ["epoch", "iteration", "dataloader", "model", "optimizer"]
            setattr(self, k, v)


class Engine(object):
    def __init__(self, custom_parser=None, config_path=None):
        logger.info("PyTorch Version {}".format(torch.__version__))
        self.state = State()
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.config = self.load_config(config_path)
        self.continue_state_object = self.args.continue_fpath

        if "WORLD_SIZE" in os.environ:
            self.distributed = int(os.environ["WORLD_SIZE"]) > 1

        if self.distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])  # self.args.local_rank
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            os.environ["MASTER_PORT"] = self.args.port
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl",
                rank=self.local_rank,
                timeout=datetime.timedelta(seconds=18000),
                world_size=self.world_size,
            )
            self.devices = [i for i in range(self.world_size)]
        else:
            self.devices = parse_devices(self.args.devices)

    def load_config(self, yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return edict(config)

    def inject_default_parser(self):
        p = self.parser
        p.add_argument("-d", "--devices", default="", help="set data parallel training")
        p.add_argument(
            "-c",
            "--resume",
            type=extant_file,
            metavar="FILE",
            dest="continue_fpath",  # 使用 dest 指定属性名称
            help="continue from one certain checkpoint",
        )
        p.add_argument("--local-rank", default=0, type=int, help="process rank on node")
        p.add_argument(
            "-p",
            "--port",
            type=str,
            default="16005",
            dest="port",
            help="port for init_process_group",
        )
        p.add_argument('--dataset_name', '-n', default='mfnet', type=str)
        # p.add_argument(
        #     "--config", type=str, required=True, help="path to the configuration file"
        # )

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        self.state.epoch = epoch
        self.state.iteration = iteration

    def save_checkpoint(self, path):
        logger.info("Saving checkpoint to file {}".format(path))
        t_start = time.time()

        state_dict = {}

        new_state_dict = OrderedDict()

        # with open('output.txt', 'w') as f:
        for k, v in self.state.model.state_dict().items():
            key = k
            # f.write(str(key) + '\n')
            if k.split(".")[0] == "module":
                key = k[7:]
            new_state_dict[key] = v
        state_dict["model"] = new_state_dict
        state_dict["optimizer"] = self.state.optimizer.state_dict()
        state_dict["epoch"] = self.state.epoch
        state_dict["iteration"] = self.state.iteration

        t_iobegin = time.time()
        torch.save(state_dict, path)
        t_end = time.time()
        logger.info(
            "Save checkpoint to file {}, "
            "Time usage:\n\tprepare checkpoint: {}, IO: {}".format(
                path, t_iobegin - t_start, t_end - t_iobegin
            )
        )

    def link_tb(self, source, target):
        ensure_dir(source)
        ensure_dir(target)
        link_file(source, target)

    def save_and_link_checkpoint(self, checkpoint_dir, log_dir, log_dir_link):
        ensure_dir(checkpoint_dir)
        if not os.path.exists(log_dir_link):
            link_file(log_dir, log_dir_link)
        current_epoch_checkpoint = os.path.join(checkpoint_dir, "epoch-{}.pth".format(self.state.epoch))
        self.save_checkpoint(current_epoch_checkpoint)
        # last_epoch_checkpoint = os.path.join(checkpoint_dir, "epoch-last.pth")
        # self.save_checkpoint(last_epoch_checkpoint)
        # link_file(current_epoch_checkpoint, last_epoch_checkpoint)

    def restore_checkpoint(self):
        t_start = time.time()
        if self.distributed:
            tmp = torch.load(self.continue_state_object, map_location=torch.device("cpu"))
        else:
            tmp = torch.load(self.continue_state_object)
        t_ioend = time.time()
        # Process the model state dictionary, removing the 'module' prefix
        model_state_dict = OrderedDict()
        for k, v in tmp["model"].items():
            new_key = k.replace("module.", "")
            model_state_dict[new_key] = v
        # self.state.model = load_model(self.state.model, tmp["model"], is_restore=True)
        self.state.model.load_state_dict(model_state_dict, strict=False)
        self.state.optimizer.load_state_dict(tmp["optimizer"])
        self.state.epoch = tmp["epoch"] + 1
        self.state.iteration = tmp["iteration"]
        del tmp
        t_end = time.time()
        logger.info(
            "Load checkpoint from file {}, "
            "Time usage:\n\tIO: {}, restore checkpoint: {}".format(
                self.continue_state_object, t_ioend - t_start, t_end - t_ioend
            )
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process"
            )
            return False
