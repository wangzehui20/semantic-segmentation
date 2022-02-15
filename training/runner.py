import os
import os.path as osp
import re
import sys
import cv2
import time
import torch
import csv
import shutil
import torch.nn.functional as F
import json
from typing import Optional, Type
import warnings
import torch.distributed as dist
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Mapping, Union
from collections import defaultdict
import numpy as np
from .callbacks import CallbackList
from torch.cuda.amp import autocast as autocast
from .pseudo_dataset import PseudoDataset
from .losses import WeightCEDiceLoss


def graytorgb(img):
    label_mapping_rgb = {0: (255, 255, 255),
                         1: (0, 0, 255),
                         2: (128, 128, 128),
                         3: (0, 128, 0),
                         4: (0, 255, 0),
                         5: (128, 0, 0),
                         6: (255, 0, 0), }
    img_rgb = np.stack(np.vectorize(label_mapping_rgb.get)(img), axis=2).astype('uint8')
    return img_rgb


def convert_label(label, inverse=False):
    label_mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        10: 8,
        11: 9,
        12: 10,
        13: 11,
        14: 12,
    }
    tmp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[tmp == k] = v
    else:
        for k, v in label_mapping.items():
            label[tmp == k] = v
        label[label > len(label_mapping) - 1] = 0
    return label


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def update_class_weights(ids, data):
    class_weights = list()
    for id in ids:
        class_weights.append(data[id])
    return np.mean(class_weights, 0)


def to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_name(instance):
    if hasattr(instance, '__name__'):
        return instance.__name__
    else:
        return to_snake(instance.__class__.__name__)


def timeit(f):
    def wrapped(*args, **kwargs):
        # start = time.time()
        res = f(*args, **kwargs)
        # print(f"{f.__name__}: {time.time() - start}")
        return res

    return wrapped


class Meter:

    def __init__(self):
        self._data = defaultdict(list)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k.find('micro') != -1:
                self._data[k] = [v.item()]
            else:
                self._data[k].append(v.item())

    def data(self, prefix=None):
        prefix = '{}_'.format(prefix) if prefix is not None else ''
        return {(prefix + name): values for name, values in self._data.items()}

    def mean(self, prefix=None):
        prefix = '{}_'.format(prefix) if prefix is not None else ''
        return {(prefix + name): sum(values) / len(values) for name, values in self._data.items()}

    def last(self, prefix=None):
        prefix = '{}_'.format(prefix) if prefix is not None else ''
        return {(prefix + name): values[-1] for name, values in self._data.items()}


class Runner:
    def __init__(
            self,
            model: torch.nn.Module,
            model_device: Union[str, torch.device],
            model_input_keys: Optional[Union[str, List[str]]] = 'image',
            model_output_keys: Optional[Union[str, List[str]]] = 'mask',
            local_rank=None,
            fp16=False,
            train_sampler=None,
            distributed=False,
            unlabeled_dataloader=None,
            pseudo_dataset=None,
            pseudo_dataloader=None,
    ):

        self.model = model
        self.device = model_device
        self.input_keys = model_input_keys if isinstance(model_input_keys, (list, tuple)) else [model_input_keys]
        self.output_keys = model_output_keys if isinstance(model_output_keys, (list, tuple)) else [
            model_output_keys]

        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.local_rank = local_rank
        self.fp16 = fp16
        self._to_device(self.model)
        self.train_sampler = train_sampler
        self.scaler = torch.cuda.amp.GradScaler()
        self.distributed = distributed
        self.unlabeled_dataloader = unlabeled_dataloader
        self.pseudo_dataset = pseudo_dataset
        self.pseudo_dataloader = pseudo_dataloader

        self.class_weights_json = load_json('/data/data/semi_compete/clip_integrate/512_128/labeled_train/class_weights.json')

    def compile(
            self,
            optimizer: Optional[Type[torch.optim.Optimizer]] = None,
            loss: Mapping[str, callable] = None,
            metrics: Mapping[str, List[callable]] = None,
    ) -> None:
        self.optimizer = optimizer
        self.loss = self._to_device(loss)
        self.metrics = self._to_device(metrics)

    @timeit
    def _to_device(self, x):

        if isinstance(x, (list, tuple)):
            return [self._to_device(xi) for xi in x]
        elif isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        else:
            if hasattr(x, 'to'):
                return x.to(self.device)
            else:
                return x

    def _model_to_mode(self, mode='train'):
        if mode == 'train' and hasattr(self.model, 'train'):
            self.model.train()
        elif mode == 'eval' and hasattr(self.model, 'eval'):
            self.model.eval()
        else:
            warnings.warn(
                "Model does not support train/eval modes, are you using traced module?",
                UserWarning,
            )

    def _prepare_input(self, batch: Mapping[str, torch.Tensor]) -> List:
        """Collect model input data from batch (collect list)"""
        if not isinstance(batch, dict):
            raise ValueError("Runner expect batches to be of type Dict! Got type {}.".format(type(batch)))
        return [batch[k] for k in batch if k in self.input_keys]

    def _prepare_output(self, model_output: Union[torch.Tensor, list, tuple, dict]) -> Mapping[str, torch.Tensor]:
        """Take model output and convert it it dict, if it is not dict"""

        if isinstance(model_output, torch.Tensor):
            model_output = [model_output]

        if isinstance(model_output, (list, tuple)):
            if len(model_output) != len(self.output_keys):
                raise ValueError("Runner have output keys {}, but model produce only {} outputs".format(
                    self.output_keys, len(model_output))
                )
            output = {k: v for k, v in zip(self.output_keys, model_output)}

        elif isinstance(model_output, dict):
            output = {k: model_output[k] for k in self.output_keys}

        else:
            raise ValueError("Model output expected to be list, dict or Tensor, got {}.".format(type(model_output)))

        return output

    # mean value of multi-gpus
    def _distributed_value(self, tensor):
        output_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensor, tensor)
        concat = torch.cat(output_tensor, dim=0)
        return torch.mean(concat)

    # create pseudo dataloader including train dataset and pseudo dataset
    def _generate_pseudo_dataloader(self, dataset, dataloader, distributed):
        pseudo_dataset = PseudoDataset(
            **dataset
        )
        pseudo_sampler = None
        if distributed:
            pseudo_sampler = torch.utils.data.distributed.DistributedSampler(pseudo_dataset)

        pseudo_dataloader = torch.utils.data.DataLoader(
            pseudo_dataset, **dataloader, sampler=pseudo_sampler,
        )
        return pseudo_dataloader

    def _update_masks(self, masks_dir):
        """
        delete and create
        :return:
        """
        dist.barrier()
        if self.local_rank == 0:
            if osp.exists(masks_dir):
                shutil.rmtree(masks_dir)
            os.makedirs(masks_dir)
        dist.barrier()

    def _check_dir(self, dir):
        dist.barrier()
        if self.local_rank == 0 and not osp.exists(dir):
            os.makedirs(dir)
        dist.barrier()

    def _reset_loss(self, class_weights):
        init_params = {
            'ignore_label': 0
        }
        init_params['class_weights'] = class_weights
        criterion = WeightCEDiceLoss(**init_params)
        return criterion

    @timeit
    def _feed_batch(self, batch) -> Mapping[str, torch.Tensor]:
        input = self._prepare_input(batch)
        output = self.model(*input)
        output = self._prepare_output(output)
        return output

    @timeit
    def _compute_losses(
            self,
            output: Mapping[str, torch.Tensor],
            target: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:

        losses_dict = {}
        class_weights = update_class_weights(target['id'], self.class_weights_json)

        # compute loss for each output
        for output_name, criterion in self.loss.items():
            criterion = self._reset_loss(class_weights)

            loss_name = 'loss_{}'.format(output_name)
            # compute auxiliary_head loss
            if 'aux' in output_name:
                losses_dict[loss_name] = self.model.module.auxiliary_head_loss_weight * criterion(
                    output[output_name], target[output_name[:-4]])
            else:
                losses_dict[loss_name] = criterion(output[output_name], target[output_name])

        # compute total loss across all outputs
        losses_dict['loss'] = sum(loss for loss in losses_dict.values())

        return losses_dict

    @timeit
    def _compute_metrics(
            self,
            output: Mapping[str, torch.Tensor],
            target: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        metrics_dict = {}
        for output_name, metrics in self.metrics.items():
            for i, metric in enumerate(metrics):
                metric_name = '{output_name}_{metric_name}'.format(
                    output_name=output_name,
                    metric_name=get_name(metric),
                )
                metric_value = metric(
                    output[output_name],
                    target[output_name],
                )
                if metric_name in metrics_dict.keys():
                    metric_name = metric_name + '_' + str(i)
                metrics_dict[metric_name] = metric_value
        return metrics_dict

    def _reset_metrics(self):
        for output_name, metrics in self.metrics.items():
            for i, metric in enumerate(metrics):
                if hasattr(metric, "reset"):
                    metric.reset()

    def _backward(self, loss: torch.Tensor, accumulation_steps: int = 1) -> None:
        total_loss = loss / accumulation_steps
        if self.fp16:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

    def _update_weights(self):
        if self.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        torch.cuda.synchronize()
        self.model.zero_grad()

    def _format_logs(self, logs):
        str_values = ['{}:{:.4f}'.format(k, v) for k, v in logs.items()]
        str_log = ', '.join(str_values)
        return str_log

    def fit(
            self,
            train_dataloader,
            train_steps=None,
            valid_dataloader=None,
            valid_steps=None,
            accumulation_steps=1,
            verbose=True,
            logdir=None,
            epochs=1,
            initial_epoch=0,
            callbacks=None,
    ) -> dict:

        if self.loss is None:
            raise ValueError('Provide loss for training!')

        # define training callbacks
        logs = {}
        callbacks = CallbackList(callbacks or [])
        callbacks.set_runner(self)
        callbacks.on_train_begin(logs=logs)

        # start training loop
        startt = time.time()
        for epoch in range(initial_epoch, epochs):
            # semi-supervised learning
            if epoch >= 200:
                self.pseudo(self.unlabeled_dataloader, epoch=epoch)
                train_dataloader = self._generate_pseudo_dataloader(self.pseudo_dataset, self.pseudo_dataloader,
                                                                    self.distributed)

            if self.train_sampler is not None:  # 必须设置随机数种子，不然不会随机
                self.train_sampler.set_epoch(epoch)
            if self.local_rank == 0:
                print('Epoch {}/{}'.format(epoch, epochs - 1), ' start_running')

            meter = Meter()
            self._reset_metrics()
            self._model_to_mode('train')

            callbacks.on_epoch_begin(epoch)

            with tqdm(total=train_steps or len(train_dataloader), desc='train', disable=not verbose, ncols=0) as pbar:
                for i, batch in enumerate(train_dataloader):
                    # batch begin callbacks
                    callbacks.on_batch_begin(i)
                    # main training process
                    batch = self._to_device(batch)
                    if self.fp16:
                        with autocast():
                            output = self._feed_batch(batch)
                            losses = self._compute_losses(output, batch)
                    else:
                        output = self._feed_batch(batch)
                        losses = self._compute_losses(output, batch)
                    self._backward(losses['loss'], accumulation_steps)
                    if (i + 1) % accumulation_steps == 0:
                        self._update_weights()

                    # collecting metrics
                    metrics = dict()
                    if self.metrics is not None:
                        metrics = self._compute_metrics(output, batch)

                    # calculate loss and metrics on all gpus
                    if self.distributed and torch.distributed.get_world_size() > 1:
                        for keys, values in losses.items():
                            losses[keys] = self._distributed_value(torch.Tensor([values]).cuda())
                        for keys, values in metrics.items():
                            metrics[keys] = self._distributed_value(torch.Tensor([values]).cuda())

                    # update batch logs
                    meter.update(**losses, **metrics)
                    batch_logs = meter.last()
                    callbacks.on_batch_end(i, batch_logs)

                    if verbose and self.local_rank == 0:
                        _logs_dict = meter.mean()
                        _logs_str = self._format_logs(_logs_dict)
                        pbar.set_postfix_str(_logs_str)
                        pbar.update()
                        # print(_logs_str)

                    if train_steps is not None and i >= train_steps - 1:
                        break

            epoch_logs = meter.mean()

            # evaluation stage
            if valid_dataloader is not None:
                elog = self.evaluate(
                    valid_dataloader,
                    steps=valid_steps,
                    verbose=verbose,
                    logdir=logdir,
                )
                epoch_logs.update(elog)

                logs[epoch] = epoch_logs

                callbacks.on_epoch_end(epoch, epoch_logs)
                if self.local_rank == 0:
                    self.writercsv(logdir, epoch_logs)

            if self.local_rank == 0:
                print('Epoch {}/{}'.format(epoch, epochs - 1), time.time() - startt)
                startt = time.time()
                print('')

        callbacks.on_train_end(logs)

        return logs

    @torch.no_grad()
    def evaluate(
            self,
            dataloader,
            steps=None,
            verbose=True,
            reduce=True,
            logdir=None,
    ):
        if self.local_rank == 0:
            savegt_path = os.path.join(logdir, 'gt')
            if os.path.exists(savegt_path) is False:
                os.makedirs(savegt_path)
            savepre_path = os.path.join(logdir, 'pre')
            if os.path.exists(savepre_path) is False:
                os.makedirs(savepre_path)

        if self.loss is None and self.metrics is None:
            raise ValueError('Provide metrics or/and losses for evaluation!')

        meter = Meter()
        self._reset_metrics()
        self._model_to_mode('eval')
        if self.local_rank == 0:
            print("val start")

        if self.local_rank == 0:
            pbar = tqdm(total=steps or len(dataloader), desc='valid', disable=not verbose, ncols=0)
        for i, batch in enumerate(dataloader):
            # if self.local_rank == 0:
            #     print("----val   ", i)

            batch = self._to_device(batch)

            if self.fp16:
                with autocast():
                    output = self._feed_batch(batch)
                    losses = self._compute_losses(output, batch) if self.loss is not None else {}
            else:
                output = self._feed_batch(batch)
                losses = self._compute_losses(output, batch) if self.loss is not None else {}

            metrics = self._compute_metrics(output, batch) if self.metrics is not None else {}

            # calculate loss and metrics on all gpus
            if torch.distributed.get_world_size() > 1:
                for keys, values in losses.items():
                    losses[keys] = self._distributed_value(torch.Tensor([values]).cuda())
                for keys, values in metrics.items():
                    metrics[keys] = self._distributed_value(torch.Tensor([values]).cuda())
            meter.update(**losses, **metrics)

            # if os.path.exists(logdir):
            #     cgt = batch['mask'].cpu().numpy()[:, 0, :, :]
            #     cgt = (cgt * 255).astype('uint8')
            #
            #     cout = output['mask'].cpu().numpy()[:, 0, :, :]
            #     cout = (cout * 255).astype('uint8')
            #
            #     for i in range(cout.shape[0]):
            #         savename = batch['id'][i]
            #
            #         filepath, _ = os.path.split(os.path.join(savegt_path, savename))
            #         if os.path.exists(filepath) is False:
            #             os.makedirs(filepath)
            #         filepath, _ = os.path.split(os.path.join(savepre_path, savename))
            #         if os.path.exists(filepath) is False:
            #             os.makedirs(filepath)
            #
            #         cv2.imwrite(os.path.join(savegt_path, savename), cgt[i])
            #         cv2.imwrite(os.path.join(savepre_path, savename), cout[i])

            if verbose and self.local_rank == 0:
                _logs_dict = meter.mean(prefix='val')
                _logs_str = self._format_logs(_logs_dict)
                pbar.set_postfix_str(_logs_str)
                pbar.update()

            if steps is not None and i >= steps - 1:
                break

        if self.local_rank == 0:
            pbar.close()

        logs = meter.mean(prefix='val') if reduce else meter.data(prefix='val')

        return logs

    @torch.no_grad()
    def pseudo(
            self,
            dataloader,
            steps=None,
            verbose=True,
            epoch=0
    ):
        self._update_masks(self.pseudo_dataset.pse_masks_dir)
        self._model_to_mode('eval')
        if self.local_rank == 0:
            print("pseudo start")

        if self.local_rank == 0:
            pbar = tqdm(total=steps or len(dataloader), desc='pseudo', disable=not verbose, ncols=0)

        for i, batch in enumerate(dataloader):
            batch = self._to_device(batch)

            if self.fp16:
                with autocast():
                    output = self._feed_batch(batch)
            else:
                output = self._feed_batch(batch)

            for output_name, output_value in output.items():
                if output_value.shape[1] == 1:
                    label = F.sigmoid(output_value)
                    soft_label = label.data.cpu().numpy().squeeze(1)
                    hard_label = (label > 0.5).data.cpu().numpy().squeeze(1).astype(np.uint8)
                else:
                    # multi class
                    label = F.softmax(output_value, dim=1)
                    soft_label = label.data.cpu().numpy()
                    hard_label = torch.argmax(label, dim=1).data.cpu().numpy().astype(np.uint8)

                for i, name in enumerate(batch['id']):
                    if np.all(hard_label[i] == 0):
                        name = 'delete_' + name

                    pseudo_label_path = osp.join(self.pseudo_dataset.pse_masks_dir, name[:-4] + '.npy')
                    np.save(pseudo_label_path, soft_label[i])

                    # hard label
                    pseudo_dir = osp.dirname(self.pseudo_dataset.pse_masks_dir)
                    cur_pseudo_dir = osp.join(pseudo_dir, 'pseudo_{}'.format(epoch))
                    self._check_dir(cur_pseudo_dir)
                    cur_pseudo_path = osp.join(cur_pseudo_dir, name[:-4] + '.png')
                    tmp_label = hard_label[i]

                    if len(soft_label.shape) == 3:
                        tmp_label[tmp_label == 1] = 255
                        cv2.imwrite(cur_pseudo_path, tmp_label)
                    else:
                        cv2.imwrite(cur_pseudo_path, convert_label(tmp_label, inverse=True))

            if verbose and self.local_rank == 0:
                pbar.update()

            if steps is not None and i >= steps - 1:
                break

        if self.local_rank == 0:
            pbar.close()

    @torch.no_grad()
    def predict(
            self,
            dataloader,
            verbose=True,
            ignore_outputs=None,
    ):
        self._model_to_mode('eval')

        ignore_outputs = ignore_outputs or []
        output = {}
        masknames = []

        with tqdm(dataloader, desc='infer',
                  disable=not verbose, file=sys.stdout) as p_dataloader:
            for i, batch in enumerate(p_dataloader):
                batch = self._to_device(batch)
                output = self._feed_batch(batch)
                masknames.extend([b['id'] for b in batch])

                for k in output.keys():
                    if k not in ignore_outputs:
                        output[k].append(
                            output[k].cpu().detach()
                        )

        output = {k: torch.cat(v, dim=0) for k, v in output.items()}
        output['id'] = masknames
        return output

    @torch.no_grad()
    def predict_on_batch(self, batch):
        batch = self._to_device(batch)
        output = self._feed_batch(batch)
        return output

    def writercsv(self, logdir, data):
        logpath = os.path.join(logdir, 'log.csv')
        firstline = []
        logline = []
        for k, v in data.items():
            if isinstance(v, float):
                v = "%.6f" % (v)
                firstline.append(k)
                logline.append(v)

        if os.path.isfile(logpath):
            with open(logpath, 'a', newline='') as f:
                csv_write = csv.writer(f, dialect='excel')
                csv_write.writerow(logline)
        else:
            with open(logpath, 'w', newline='') as f:
                csv_write = csv.writer(f, dialect='excel')
                csv_write.writerow(firstline)
                csv_write.writerow(logline)


class GPUNormRunner(Runner):

    def _prepare_input(self, batch):
        # image = batch["image"]
        # image -= torch.tensor([123.675, 116.28, 103.53], device=self.device).reshape(1, 3, 1, 1)
        # image /= torch.tensor([58.395, 57.12, 57.375], device=self.device).reshape(1, 3, 1, 1)
        # batch["image"] = image
        return super()._prepare_input(batch)
