# Reimplementaion of training loop
import torch
import torch.nn as nn
import torch.optim as optim
import glob, os
import numpy as np

import logging
from torch.optim import Adam
from collections import defaultdict
from datetime import datetime
import pandas as pd
import json
from time import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tensorboardX.writer import SummaryWriter


import shutil
import sys


'''
Original Code:  https://github.com/sweetcocoa/PinkBlack/tree/a9677cad081083b5d466b5f5116d4d93c3e874b3

Reimplementation by: https://github.com/athanatos96
'''



def padding(arg, width, pad=' '):
    if isinstance(arg, float):
        return '{:.6f}'.format(arg).center(width, pad)
    elif isinstance(arg, int):
        return '{:6d}'.format(arg).center(width, pad)
    elif isinstance(arg, str):
        return arg.center(width, pad)
    elif isinstance(arg, tuple):
        if len(arg) != 2:
            raise ValueError('Unknown type: {}'.format(type(arg), arg))
        if not isinstance(arg[1], str):
            raise ValueError('Unknown type: {}'
                             .format(type(arg[1]), arg[1]))
        return padding(arg[0], width, pad=pad)
    else:
        raise ValueError('Unknown type: {}'.format(type(arg), arg))
    
def print_row(kwarg_list=[], pad=' '):
    len_kwargs = len(kwarg_list)
    term_width = shutil.get_terminal_size().columns
    width = min((term_width - 1 - len_kwargs) * 9 // 10, 150) // len_kwargs
    row = '|{}' * len_kwargs + '|'
    columns = []
    for kwarg in kwarg_list:
        columns.append(padding(kwarg, width, pad=pad))
    print(row.format(*columns))

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_accuracy(pred, target):
    pred = torch.max(pred, 1)[1]
    corrects = torch.sum(pred == target).float()
    return corrects / pred.size(0)


class Trainer2:
    experiment_name = None

    def __init__(
        self,
        net,
        criterion=None,
        metric=None,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        optimizer=None,
        lr_scheduler=None,
        tensorboard_dir="./pinkblack_tb/",
        ckpt="./ckpt/ckpt.pth",
        experiment_id=None,
        clip_gradient_norm=False,
        is_data_dict=False,
    ):
        self.net = net
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        if metric is None:
            print("No metric selected")
        self.metric = metric
        
        self.dataloader = dict()
        if train_dataloader is not None:
            self.dataloader["train"] = train_dataloader
        if val_dataloader is not None:
            self.dataloader["val"] = val_dataloader
        if test_dataloader is not None:
            self.dataloader["test"] = test_dataloader

        if train_dataloader is None or val_dataloader is None:
            logging.warning("Init Trainer :: Two dataloaders are needed!")

        self.optimizer = (
            Adam(filter(lambda p: p.requires_grad, self.net.parameters()))
            if optimizer is None
            else optimizer
        )
        
        self.lr_scheduler = lr_scheduler

        self.ckpt = ckpt

        self.config = defaultdict(float)
        self.config["max_train_metric"] = -1e8
        self.config["max_val_metric"] = -1e8
        self.config["max_test_metric"] = -1e8
        self.config["tensorboard_dir"] = tensorboard_dir
        self.config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config["clip_gradient_norm"] = clip_gradient_norm
        self.config["is_data_dict"] = is_data_dict
        
        if experiment_id is None:
            self.config["experiment_id"] = self.config["timestamp"]
        else:
            self.config["experiment_id"] = experiment_id

        self.dataframe = pd.DataFrame()

        self.device = Trainer2.get_model_device(self.net)
        if self.device == torch.device("cpu"):
            logging.warning(
                "Init Trainer :: Do you really want to train the network on CPU instead of GPU?"
            )
            
        if self.config["tensorboard_dir"] is not None:
            self.tensorboard = SummaryWriter(self.config["tensorboard_dir"])
        else:
            self.tensorboard = None

        self.callbacks = defaultdict(list)
    
    def save(self, f=None):
        if f is None:
            f = self.ckpt
        os.makedirs(os.path.dirname(f), exist_ok=True)
        if isinstance(self.net, nn.DataParallel):
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        torch.save(state_dict, f)
        torch.save(self.optimizer.state_dict(), f + ".optimizer")

        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), f + ".scheduler")

        with open(f + ".config", "w") as fp:
            json.dump(self.config, fp)

        self.dataframe.to_csv(f + ".csv", float_format="%.6f", index=False)
    
    def load(self, f=None):
        if f is None:
            f = self.ckpt

        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(torch.load(f, map_location=self.device))
        else:
            self.net.load_state_dict(torch.load(f, map_location=self.device))

        if os.path.exists(f + ".config"):
            with open(f + ".config", "r") as fp:
                dic = json.loads(fp.read())
            self.config = defaultdict(float, dic)
            print("Loaded,", self.config)

        if os.path.exists(f + ".optimizer"):
            self.optimizer.load_state_dict(torch.load(f + ".optimizer"))

        if os.path.exists(f + ".scheduler") and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(torch.load(f + ".scheduler"))

        if os.path.exists(f + ".csv"):
            self.dataframe = pd.read_csv(f + ".csv")

        if self.config["tensorboard_dir"] is not None:
            self.tensorboard = SummaryWriter(self.config["tensorboard_dir"])
        else:
            self.tensorboard = None
    
    def train(
        self, epoch=None, phases=None, step=None, validation_interval=1, save_every_validation=False
    ):
        """
        :param epoch: train dataloader를 순회할 횟수
        :param phases: ['train', 'val', 'test'] 중 필요하지 않은 phase를 뺄 수 있다.
        >> trainer.train(1, phases=['val'])
        :param step: epoch이 아닌 step을 훈련단위로 할 때의 총 step 수.
        :param validation_interval: validation 간격
        :param save_every_validation: True이면, validation마다 checkpoint를 저장한다.
        :return: None
        """
        
        print("Start of Training")
        
        if phases is None:
            phases = list(self.dataloader.keys())

        if epoch is None and step is None:
            raise ValueError("PinkBlack.trainer :: epoch or step should be specified.")

        train_unit = "epoch" if step is None else "step"
        self.config[train_unit] = int(self.config[train_unit])

        num_unit = epoch if step is None else step
        validation_interval = 1 if validation_interval <= 0 else validation_interval

        kwarg_list = [train_unit]
        for phase in phases:
            kwarg_list += [f"{phase}_loss", f"{phase}_metric"]
        kwarg_list += ["lr", "time"]

        print_row(kwarg_list=[""] * len(kwarg_list), pad="-")
        print_row(kwarg_list=kwarg_list, pad=" ")
        print_row(kwarg_list=[""] * len(kwarg_list), pad="-")

        start = self.config[train_unit]

        for i in range(start, start + num_unit, validation_interval):
            start_time = time()
            if train_unit == "epoch":
                for phase in phases:
                    self.config[f"{phase}_loss"], self.config[f"{phase}_metric"] = self._train(
                        phase, num_steps=len(self.dataloader[phase])
                    )
                    for func in self.callbacks[phase]:
                        func()
                self.config[train_unit] += 1
            elif train_unit == "step":
                for phase in phases:
                    if phase == "train":
                        # num_unit 이 validation interval로 나눠떨어지지 않는 경우
                        num_steps = min((start + num_unit - i), validation_interval)
                        self.config[train_unit] += num_steps
                    else:
                        num_steps = len(self.dataloader[phase])
                    self.config[f"{phase}_loss"], self.config[f"{phase}_metric"] = self._train(
                        phase, num_steps=num_steps
                    )
                    for func in self.callbacks[phase]:
                        func()
            else:
                raise NotImplementedError
            
            #print("After_train")
            
            # Update Scheduler
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(self.config["val_metric"])
                else:
                    self.lr_scheduler.step()
            
            
            i_str = str(self.config[train_unit])
            is_best = self.config["max_val_metric"] < self.config["val_metric"]
            if is_best:
                for phase in phases:
                    self.config[f"max_{phase}_metric"] = max(
                        self.config[f"max_{phase}_metric"], self.config[f"{phase}_metric"]
                    )
                i_str = (str(self.config[train_unit])) + "-best"
                
            
            elapsed_time = time() - start_time
            if self.tensorboard is not None:
                _loss, _metric = {}, {}
                for phase in phases:
                    _loss[phase] = self.config[f"{phase}_loss"]
                    _metric[phase] = self.config[f"{phase}_metric"]

                self.tensorboard.add_scalars(
                    f"{self.config['experiment_id']}/loss", _loss, self.config[train_unit]
                )
                self.tensorboard.add_scalars(
                    f"{self.config['experiment_id']}/metric", _metric, self.config[train_unit]
                )
                self.tensorboard.add_scalar(
                    f"{self.config['experiment_id']}/time", elapsed_time, self.config[train_unit]
                )
                self.tensorboard.add_scalar(
                    f"{self.config['experiment_id']}/lr",
                    self.optimizer.param_groups[0]["lr"],
                    self.config[train_unit],
                )
            
            print_kwarg = [i_str]
            for phase in phases:
                print_kwarg += [self.config[f"{phase}_loss"], self.config[f"{phase}_metric"]]
            print_kwarg += [self.optimizer.param_groups[0]["lr"], elapsed_time]
            
            print_row(kwarg_list=print_kwarg, pad=" ")
            print_row(kwarg_list=[""] * len(kwarg_list), pad="-")
            self.dataframe = self.dataframe.append(
                dict(zip(kwarg_list, print_kwarg)), ignore_index=True
            )

            if is_best:
                self.save(self.ckpt)
                if Trainer2.experiment_name is not None:
                    self.update_experiment()

            if save_every_validation:
                self.save(self.ckpt + f"-{self.config[train_unit]}")
        print("END of loop")
            
    def _step(self, phase, iterator, only_inference=False):
        #print("Inside _Step")
        #print(type(self.config["is_data_dict"]))
        
        if self.config["is_data_dict"]:
            #print("Inside if self.config[is_data_dict]")
            #print("1a")
            batch_dict = next(iterator)
            #print("2a")
            batch_size = batch_dict[list(batch_dict.keys())[0]].size(0)
            #print("3a")
            for k, v in batch_dict.items():
                #print("4a")
                batch_dict[k] = v.to(self.device)
                #print("5a")
            #print("6a")
        else:
            batch_x, batch_y = next(iterator)
            if isinstance(batch_x, list):
                batch_x = [x.to(self.device) for x in batch_x]
            else:
                batch_x = [batch_x.to(self.device)]

            if isinstance(batch_y, list):
                batch_y = [y.to(self.device) for y in batch_y]
            else:
                batch_y = [batch_y.to(self.device)]

            batch_size = batch_x[0].size(0)
            
        #print("7a")    
            
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(phase == "train"):
            #print("8a") 
            if self.config["is_data_dict"]:
                #print("9a") 
                outputs = self.net(batch_dict)
                #print("outputs: ", outputs.shape)
                #print("10a") 
                if not only_inference:
                    #print("11a") 
                    loss = self.criterion(outputs, batch_dict)
                    #print("12a") 
            else:
                outputs = self.net(*batch_x)
                if not only_inference:
                    loss = self.criterion(outputs, *batch_y)
            #print("13a") 
            if only_inference:
                return outputs
            #print("14a") 
            #print("Phase ", phase)
            if phase == "train":
                #print("15a")
                loss.backward()
                #print("16a")
                #print("self.config[clip_gradient_norm]: ", self.config["clip_gradient_norm"] )
                if self.config["clip_gradient_norm"]:
                    #print("17a")
                    clip_grad_norm_(self.net.parameters(), self.config["clip_gradient_norm"])
                    #print("18a")
                self.optimizer.step()
                #print("19a")
            #print("20a")
        #print("21a")    
        with torch.no_grad():
            #print("22a")
            if self.config["is_data_dict"]:
                #print("23a")
                metric = self.metric(outputs, batch_dict)
                #print("24a")
            else:
                #print("23b")
                metric = self.metric(outputs, *batch_y)
                #print("24b")
            #print("25a")
        #print("26a")
        return {"loss": loss.item(), "batch_size": batch_size, "metric": metric.item()}
    
    def _train(self, phase, num_steps=0):
        #print(f"Inside _train, with phase: {phase}, num_steps: {num_steps}")
        
        
        running_loss = AverageMeter()
        running_metric = AverageMeter()

        #print("1")
        
        if phase == "train":
            self.net.train()
        else:
            self.net.eval()

        #print("2")
        
        dataloader = self.dataloader[phase]
        step_iterator = iter(dataloader)
        tq = tqdm(range(num_steps), leave=False)
        #print("3")
        for st in tq:
            #print("4")
            if (st + 1) % len(dataloader) == 0:
                step_iterator = iter(dataloader)
            #print("5")
            results = self._step(phase=phase, iterator=step_iterator)
            #print("6")
            tq.set_description(f"Loss:{results['loss']:.4f}, Metric:{results['metric']:.4f}")
            #print("7")
            running_loss.update(results["loss"], results["batch_size"])
            #print("8")
            running_metric.update(results["metric"], results["batch_size"])
            #print("9")
        #print("End _train")
        return running_loss.avg, running_metric.avg
    
    @staticmethod
    def get_model_device(net):
        device = torch.device("cpu")
        for param in net.parameters():
            device = param.device
            break
        return device

    @staticmethod
    def set_experiment_name(name):
        Trainer2.experiment_name = name