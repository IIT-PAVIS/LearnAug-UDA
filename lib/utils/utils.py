import os
import sys
import yaml
import json
import logging
import time
import torch

def setup_logger(log_dir=None, resume=False):
    plain_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S'
    )
    logger = logging.getLogger() # root logger
    logger.setLevel(logging.INFO)
    s_handler = logging.StreamHandler(stream=sys.stdout)
    s_handler.setFormatter(plain_formatter)
    s_handler.setLevel(logging.INFO)
    logger.addHandler(s_handler)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        if not resume and os.path.exists(os.path.join(log_dir, 'console.log')):
            os.remove(os.path.join(log_dir, 'console.log'))
        f_handler = logging.FileHandler(os.path.join(log_dir, 'console.log'))
        f_handler.setFormatter(plain_formatter)
        f_handler.setLevel(logging.INFO)
        logger.addHandler(f_handler)


class AvgMeter:
    def __init__(self, ema_coef=0.9):
        self.ema_coef = ema_coef
        self.ema_params = {}
        self.sum_params = {}
        self.counter = {}

    def add(self, params:dict, ignores:list = []):
        for k, v in params.items():
            if k in ignores:
                continue
            if not k in self.ema_params.keys():
                self.ema_params[k] = v
                self.counter[k] = 1
            else:
                self.ema_params[k] -= (1 - self.ema_coef) * (self.ema_params[k] - v)
                self.counter[k] += 1
            if not k in self.sum_params.keys():
                self.sum_params[k] = v
            else:
                self.sum_params[k] += v

    def state(self, header="", footer="", ignore_keys=None):
        if ignore_keys is None:
            ignore_keys = set()
        state = header
        for k, v in self.ema_params.items():
            if k in ignore_keys:
                continue
            state += f" {k} {v:.6g} |"
        return state + " " + footer

    def mean_state(self, header="", footer=""):
        epoch_dict = {}
        state = header
        for k, v in self.sum_params.items():
            state += f" {k} {v/self.counter[k]:.6g} |"
            epoch_dict[k] = v/self.counter[k]
            self.counter[k] = 0
        state += footer

        self.sum_params = {}

        return state, epoch_dict

    def reset(self):
        self.ema_params = {}
        self.sum_params = {}
        self.counter = {}
