# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
from typing import Tuple

from copy import deepcopy
from datetime import datetime
import json
import os
from pprint import pprint
import shlex
import sys
from time import time
import traceback
import warnings

import numpy as np
import PIL.Image
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
from torchmetrics import MeanMetric
from tqdm import tqdm as tqdm_
import wandb

from nicr_mt_scene_analysis.checkpointing import CheckpointHelper
from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_mt_scene_analysis.data import infer_batch_size
from nicr_mt_scene_analysis.logging import CSVLogger
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model
from nicr_mt_scene_analysis.utils import cprint
from nicr_mt_scene_analysis.utils import cprint_step

# internal stuff, if not available, use fallbacks to ensure that the script
# runs without these dependencies
try:

    from nicr_cluster_utils.datasets import load_dataset
    from nicr_cluster_utils.utils import get_job_id
    from nicr_cluster_utils.utils.wandb_integration import is_wandb_available
    from nicr_cluster_utils.utils import rename_job

except ImportError:

    _return_none = lambda *args, **kwargs: None
    load_dataset = _return_none
    rename_job = _return_none
    get_job_id = _return_none

    is_wandb_available = lambda *args, **kwargs: True

from dveformer.args import ArgParserDVEFormer
from dveformer.data import cast_to_dtype
from dveformer.data import get_datahelper
from dveformer.data import parse_datasets
from dveformer.linear_probing import DVEFormerLinearProbing
from dveformer.linear_probing import get_linear_probing_task_helper
from dveformer.loss_weighting import get_loss_weighting_module
from dveformer.lr_scheduler import get_lr_scheduler
from dveformer.model import DVEFormer
from dveformer.optimizer import get_optimizer
from dveformer.preprocessing import get_preprocessor
from dveformer.task_helper import get_task_helpers
from dveformer.task_helper import TaskHelperType
from dveformer.visualization import setup_shared_color_generators
from dveformer.visualization import visualize
from dveformer.weights import load_weights


class RunHelper:
    def __init__(
        self,
        args,
        model: DVEFormer,
        task_helpers: Tuple[TaskHelperType],
        device: torch.device,
    ) -> None:
        super().__init__()
        # store args to have them later
        self.args = args

        self.model = model.to(device)

        self._task_helpers = task_helpers
        for task_helper in self._task_helpers:
            task_helper.initialize(device)

        # some internal stuff
        self._device = device
        self._validation_best_metrics_cache = {}
        self._accumulated_step_metrics = {}

        # loss weighting
        self._loss_weighting_module = get_loss_weighting_module(args)
        self._loss_amp = torch.float32
        if args.loss_amp == 'fp16':
            self._loss_amp = torch.float16
        elif args.loss_amp == 'bfp16':
            self._loss_amp = torch.bfloat16

    def reset(self):
        # perform internal reset (e.g., after performing a sanity check)
        # reset loss weights
        self._loss_weighting_module.reset_weights()

        # reset internal caches
        self._validation_best_metrics_cache = {}
        self._accumulated_step_metrics = {}

    def _update_accumulated_step_metrics(self, logs, batch_size):
        metrics = self._accumulated_step_metrics    # pep8
        for key, value in logs.items():
            # create metric object if it does not yet exist
            if key not in metrics:
                metrics[key] = MeanMetric().to(self._device)
            # update metric
            metrics[key].update(value, weight=batch_size)

    def set_training_mode(self) -> None:
        torch.set_grad_enabled(True)
        self.model.train()

    def set_inference_mode(self) -> None:
        torch.set_grad_enabled(False)
        self.model.eval()

    def training_step(self, batch, batch_idx):
        assert self.model.training

        # apply model
        batch = move_batch_to_device(batch, device=self._device)
        predictions_post = self.model(batch, do_postprocessing=True)

        # apply task helpers
        losses = {}
        logs = {}

        # Convert to correct datatype if required
        if self.args.decoder_amp != self.args.loss_amp:
            predictions_post = cast_to_dtype(predictions_post, self._loss_amp)

        # Setup loss amp
        with torch.amp.autocast(
            device_type=self.args.device,
            enabled=self._loss_amp != torch.float32,
            dtype=self._loss_amp
        ):
            for task_helper in self._task_helpers:
                task_loss_dict, task_logs = task_helper.training_step(
                    batch=batch,
                    batch_idx=batch_idx,
                    predictions_post=predictions_post
                )
                losses.update(task_loss_dict)
                logs.update(task_logs)

            # accumulate losses
            loss = self._loss_weighting_module.reduce_losses(losses, batch_idx)

        # add total loss to logs
        logs['total_loss'] = loss.detach().clone()

        # update accumulated step metrics
        self._update_accumulated_step_metrics(
            logs={f'train_{key}': value for key, value in logs.items()},
            batch_size=infer_batch_size(batch)
        )

        return loss

    def training_get_artifacts_and_metrics(self):
        artifacts, metrics = {}, {}

        # handle accumulated step metrics
        for key, metric in self._accumulated_step_metrics.items():
            if 'train' not in key:
                continue
            metrics[key] = metric.compute()
            # reset metric to be ready for next epoch
            metric.reset()

        return artifacts, metrics

    def validation_step(self, batch, batch_idx):
        assert not self.model.training
        # apply model
        batch = move_batch_to_device(batch, device=self._device)
        predictions_post = self.model(batch, do_postprocessing=True)

        # apply task helpers
        losses = {}
        logs = {}
        for task_helper in self._task_helpers:
            task_loss_dict, task_logs = task_helper.validation_step(
                batch=batch,
                batch_idx=batch_idx,
                predictions_post=predictions_post
            )
            losses.update(task_loss_dict)
            logs.update(task_logs)

        # accumulate losses
        loss = self._loss_weighting_module.reduce_losses(losses, batch_idx)

        # add total loss to logs
        logs['total_loss'] = loss.detach().clone()

        # update accumulated step metrics
        self._update_accumulated_step_metrics(
            logs={f'valid_{key}': value for key, value in logs.items()},
            batch_size=infer_batch_size(batch)
        )

        return loss, predictions_post

    def validation_get_artifacts_examples_metrics(self):
        artifacts, examples, metrics = {}, {}, {}

        # handle accumulated step metrics
        for key, metric in self._accumulated_step_metrics.items():
            if 'valid' not in key:
                continue
            metrics[key] = metric.compute()
            # reset metric to be ready for next epoch
            metric.reset()

        # apply task helpers
        for task_helper in self._task_helpers:
            task_result = task_helper.validation_epoch_end()
            task_artifacts, task_examples, task_logs = task_result
            metrics.update({f'valid_{key}': value
                            for key, value in task_logs.items()})
            artifacts.update({f'valid_{key}': value
                              for key, value in task_artifacts.items()})
            examples.update({f'valid_{key}': value
                             for key, value in task_examples.items()})

        # update cache for currently best metrics
        def force_tensor(v):
            return v if isinstance(v, torch.Tensor) else torch.tensor(v)

        cache = self._validation_best_metrics_cache
        for key in metrics:
            # determine behavior
            if any(m in key for m in ('miou', 'acc', 'rq', 'sq', 'pq')):
                fn = torch.greater
                default = torch.tensor(-torch.inf)
            elif 'mae' in key or 'rmse' in key:
                fn = torch.less
                default = torch.tensor(torch.inf)
            else:
                continue

            # add or update entry in cache
            key_best = f'{key}_best'
            value_cur = metrics[key]
            value_best = cache.get(key_best, default)
            if fn(force_tensor(value_cur), force_tensor(value_best)).item():
                cache[key_best] = value_cur

        # add best metrics to current logs
        metrics.update(cache)

        return artifacts, examples, metrics


def main():
    # Args & General Stuff -----------------------------------------------------
    parser = ArgParserDVEFormer()
    args = parser.parse_args()

    if args.enable_tf32:
        # Enables TensorFloat32 for matmul operations if supported by the
        # hardware for faster training.
        torch.set_float32_matmul_precision('high')

    if args.disable_progress_bars:
        # dummy tqdm function that only prints the description and step number
        def tqdm(obj, **kwargs):
            if 'desc' in kwargs:
                print(kwargs['desc'],
                      f"({kwargs.get('total', 'unknown number of')} steps)")
            return obj
    else:
        # use tqdm
        tqdm = tqdm_

    # get dataset path via nicr-cluster-utils
    if args.dataset_path is None:
        dataset_paths = []
        for ds in parse_datasets(args.dataset):
            ds_name = ds['name']
            if ds_name in ('scenenetrgbd',):
                cluster_ds_name = f'nicr-scene-analysis-datasets-{ds_name}-v054'
            elif ds_name in ('cityscapes',):
                cluster_ds_name = f'nicr-scene-analysis-datasets-{ds_name}-v050'
            else:
                cluster_ds_name = f'nicr-scene-analysis-datasets-{ds_name}-v081'

            ds_path = str(load_dataset(cluster_ds_name, mount=True, do_not_copy_to_local=True))
            dataset_paths.append(ds_path)

        args.dataset_path = ':'.join(dataset_paths)

    # prepare results paths
    if not args.is_resumed_training:
        starttime = datetime.now().strftime('%Y_%m_%d-%H_%M_%S-%f')
        results_path = os.path.abspath(os.path.join(
            args.results_basepath,
            '_debug_runs' if args.debug else '',
            args.dataset.replace(':', '+'),
            f'run_{starttime}'
        ))
    else:
        # write results to same folder as in previous training
        results_path = args.resume_path
    os.makedirs(results_path, exist_ok=args.is_resumed_training)
    artifacts_path = os.path.join(results_path, 'artifacts')
    os.makedirs(artifacts_path, exist_ok=args.is_resumed_training)
    checkpoints_path = os.path.join(results_path, 'checkpoints')
    os.makedirs(checkpoints_path, exist_ok=args.is_resumed_training)
    examples_path = os.path.join(results_path, 'examples')
    os.makedirs(examples_path, exist_ok=args.is_resumed_training)
    print(f"Writing results to '{results_path}'.")

    # append some information to args
    args.job_id = get_job_id()
    args.results_path = results_path
    args.artifacts_path = artifacts_path
    args.checkpoints_path = checkpoints_path
    args.examples_path = examples_path
    args.start_timestamp = int(time())

    if not args.validation_only:
        # set up wandb

        # convert tuples/lists to let them appear in parallel coordinate plots
        w_args = deepcopy(args)
        for k, v in dict(vars(w_args)).items():
            if isinstance(v, (list, tuple)):
                v_str = ', '.join(str(v_) for v_ in v)
                if not isinstance(v[0], str):
                    # prepend 's ' to make sure wandb handles it correctly
                    v_str = f's {v_str}'
                setattr(w_args, f'{k}_str', v_str)

        if not is_wandb_available():
            print("Weights & Biases is not available, forcing offline mode.")
            args.wandb_mode = 'offline'

        wandb.init(
            dir=results_path,
            entity='nicr',
            config=w_args,
            mode=args.wandb_mode,
            project=args.wandb_project,
            settings=wandb.Settings(start_method='fork')
        )
        # set epoch as default x axis
        wandb.run.define_metric('epoch')
        wandb.run.define_metric("*", step_metric='epoch', step_sync=True)

        # append some information to args
        args.wandb_name = wandb.run.name
        args.wandb_id = wandb.run.id
        args.wandb_url = wandb.run.url

        # rename job on cluster
        rename_job(wandb.run.name)

        # dump args ------------------------------------------------------------
        if not args.is_resumed_training:
            # argv only if not resuming
            with open(os.path.join(args.results_path, 'argsv.txt'), 'w') as f:
                f.write(shlex.join(sys.argv))
                f.write('\n')

        with open(os.path.join(results_path, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

    # Data & Model -------------------------------------------------------------
    cprint_step("Get model and dataset")
    # get datahelper
    data = get_datahelper(args)

    if args.weights_filepath is not None:
        args.no_pretrained_backbone = True

    # get model
    if not args.enable_linear_probing:
        model = DVEFormer(args, dataset_config=data.dataset_config)
    else:
        print("Linear probing enabled: keeping the base model frozen and optimizing a linear head.")
        model = DVEFormerLinearProbing(
            args=args, dataset_config=data.dataset_config,
        )

    # load weights (account for renamed or missing keys, specific dataset
    # combinations, pretraining configurations)
    if args.weights_filepath is not None:
        print(f"Loading (pretrained) weights from: '{args.weights_filepath}'.")
        checkpoint = torch.load(args.weights_filepath,
                                map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        if 'epoch' in checkpoint:
            print(f"-> Epoch: {checkpoint['epoch']}")
        if args.debug and 'logs' in checkpoint:
            print("-> Logs/Metrics:")
            pprint(checkpoint['logs'])

        load_weights(args, model, state_dict, verbose=True)

    if args.compile_model:
        model.compile()

    # set preprocessor to datasets (note, preprocessing depends on model)
    downscales = set()
    for decoder in model.decoders.values():
        downscales |= set(decoder.side_output_downscales)

    # prepare downscales for validation
    valid_downscales = set()
    if args.debug:
        valid_downscales = set(downscales)

    # If less upsamplings are used, downscales are still required for
    # loss computation, even if no multi-scale supervision is used.
    # 2 upsamplings are default, to get from H/4 x W/4 resoulution after
    # decoder blocks to input resolution.
    # If fewer upsampling modules are used, we need the single additional
    # downscaling to still be able to compute the loss at the correct.
    # TODO: Bad default?
    if args.dense_visual_embedding_decoder_n_upsamplings is not None:
        if args.dense_visual_embedding_decoder_n_upsamplings < 2:
            downscale = 2 ** (2 - args.dense_visual_embedding_decoder_n_upsamplings)
            downscales |= set([downscale])
            # Also use downscale for validation
            valid_downscales |= set([downscale])

    data.set_train_preprocessor(
        get_preprocessor(
            args,
            dataset=data.dataset_train,
            phase='train',
            multiscale_downscales=tuple(downscales)
        )
    )

    data.set_valid_preprocessor(
        get_preprocessor(
            args,
            dataset=data.datasets_valid[0],
            phase='test',
            multiscale_downscales=tuple(valid_downscales)
        )
    )

    # export onnx model to be able to debug the model's structure
    if args.debug:
        cprint_step("Export ONNX model")
        # use 'EXPORT_ONNX_MODELS=true python ...' to export the model
        from torch.onnx import TrainingMode

        # get some valid data
        batch = next(iter(data.train_dataloader))
        batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        fp = os.path.join(results_path, 'model.onnx')
        # TODO: export for Dropout2D (feature_dropout) to enable mode PRESERVE
        if export_onnx_model(fp, model, (batch, {}),
                             training_mode=TrainingMode.EVAL,
                             force_export=False,
                             use_fallback=True):
            print(f"Wrote ONNX model to '{fp}'.")
        else:
            print("Export skipped. Set `EXPORT_ONNX_MODELS=true` to enable.")

    # Training Stuff -----------------------------------------------------------
    # logging (note, appends to existing metrics file)
    csv_logger = CSVLogger(filepath=os.path.join(results_path, 'metrics.csv'),
                           write_interval=1)

    # optimizer and lr scheduler
    if not args.enable_linear_probing:
        optimizer = get_optimizer(args, model.parameters())
    else:
        optimizer = get_optimizer(args, model.linear_probing_head.parameters())
    lr_scheduler = get_lr_scheduler(args, optimizer)
    scaler = None
    if set([args.encoder_amp, args.context_module_amp, args.decoder_amp]) != {'disabled'}:
        # Enable scaler to improve numerical stability
        scaler = torch.amp.GradScaler()

    # get task helper
    task_helpers = list(get_task_helpers(args, data.dataset_train))
    if args.enable_linear_probing:
        task_helpers.append(
            get_linear_probing_task_helper(
                args=args,
                dataset=data.dataset_train,
                model=model
            )
        )
    task_helpers = tuple(task_helpers)

    # wrap model in run helper
    run = RunHelper(
        args,
        model=model,
        task_helpers=task_helpers,
        device=torch.device(args.device)
    )

    # check for resumed training
    if args.resume_ckpt_filepath is not None:
        cprint_step("Resume training")
        checkpoint = torch.load(args.resume_ckpt_filepath,
                                map_location=torch.device('cpu'))
        print(f"Checkpoint: '{args.resume_ckpt_filepath}'")
        next_epoch = checkpoint['epoch'] + 1
        print(f"Last epoch: {checkpoint['epoch']}, next epoch: {next_epoch}")
        print("Replacing state dicts for model, optimizer, and lr scheduler.")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        if args.enable_linear_probing:
            head_state = checkpoint.get('linear_probing_head_state_dict')
            if head_state is not None:
                model.linear_probing_head.load_state_dict(head_state)
    else:
        # training starts from scratch
        next_epoch = 0

    # checkpointing
    if args.checkpointing_metrics is None:
        warnings.warn(
            "No checkpoints will be saved. Please provide the metrics by which "
            "you want to checkpoint the model weights with "
            "`--checkpointing-metrics`."
        )
    checkpoint_helper = CheckpointHelper(
        metric_names=args.checkpointing_metrics,
        debug=True    # args.debug
    )

    # Simple Sanity Check ------------------------------------------------------
    if not args.skip_sanity_check:
        # ensure that crucial parts (data, forward, metrics, ...) are working
        # as expected, the check is done by forwarding a single batch of all
        # dataloaders WITHOUT backpropagation.
        cprint_step("Perform sanity check")

        # disable forward stats tracking (e.g., batchnorm)
        for m in model.modules():
            if hasattr(m, 'track_running_stats'):
                m.track_running_stats = False

        # check training (single batch)
        batch = next(iter(data.train_dataloader))
        assert isinstance(run.training_step(batch, 0), torch.Tensor)
        assert run.training_get_artifacts_and_metrics()

        # re-enable forward stats tracking (e.g., batchnorm)
        for m in model.modules():
            if hasattr(m, 'track_running_stats'):
                m.track_running_stats = True

        # check validation (single batch for all valid sets)
        run.set_inference_mode()
        for valid_dataloader in data.valid_dataloaders:
            batch = next(iter(valid_dataloader))
            validation_result, _ = run.validation_step(batch, 0)
            assert isinstance(validation_result, torch.Tensor)
        result = run.validation_get_artifacts_examples_metrics()  # also resets
        assert result

        # check metrics for checkpointing
        artifacts, examples, metrics = result
        for ckpt_metric in args.checkpointing_metrics or []:
            assert checkpoint_helper._determine_checkpoint_metrics(
                ckpt_metric, metrics
            )

        # reset run helper states (loss weighting module and metric caches)
        run.reset()

        # everything seems to work
        print("Fine.")

    # Validation ---------------------------------------------------------------
    if args.validation_only:
        cprint_step("Run validation only")

        if args.visualize_validation:
            print("Writing visualizations to: "
                  f"'{args.visualization_output_path}'.")
            os.makedirs(args.visualization_output_path, exist_ok=True)

            # dump args
            with open(os.path.join(args.visualization_output_path,
                                   'args.json'), 'w') as f:
                json.dump(vars(args), f, sort_keys=True, indent=4)
            with open(os.path.join(args.visualization_output_path,
                                   'argsv.txt'), 'w') as f:   # should be argv
                f.write(shlex.join(sys.argv))
                f.write('\n')

        # use shared color generators to ensure consistent colors and to speed
        # up visualization
        setup_shared_color_generators(data.dataset_train.config)

        run.set_inference_mode()
        batch_idx = 0
        for i, valid_dataloader in enumerate(data.valid_dataloaders):
            tqdm_desc = f'Validation {i+1}/{len(data.valid_dataloaders)}'
            tqdm_desc += f' ({valid_dataloader.dataset.camera})'
            for batch in tqdm(valid_dataloader,
                              total=len(valid_dataloader),
                              desc=tqdm_desc):
                _, predictions = run.validation_step(batch, batch_idx)
                if args.visualize_validation:
                    output_path = os.path.join(
                        args.visualization_output_path,
                        args.validation_split.replace(':', '+')
                    )
                    visualize(
                        output_path=output_path,
                        batch=batch,
                        predictions=predictions,
                        dataset_config=data.dataset_train.config
                    )

                batch_idx += 1

        # get and print validation metrics
        _, _, metrics = run.validation_get_artifacts_examples_metrics()
        print("Validation results:")
        pprint(metrics)

        # stop here
        return

    # Training -----------------------------------------------------------------
    cprint_step("Start training")
    # overfitting
    if args.overfit_n_batches > 0:
        # force overfitting (training+validation) to overfit_n_batches batches
        # of the valid set
        data.enable_overfitting_mode(n_valid_batches=args.overfit_n_batches)

    # training loop
    try:
        for epoch in range(next_epoch, args.n_epochs):
            cprint(f"Epoch: {epoch:04d}/{args.n_epochs-1:04d}",
                   color='cyan', attrs=('bold',))
            epoch_logs = {'epoch': epoch, 'lr': lr_scheduler.get_last_lr()[0]}

            # training
            run.set_training_mode()
            for batch_idx, batch in tqdm(enumerate(data.train_dataloader),
                                         total=len(data.train_dataloader),
                                         desc='Training'):
                loss = run.training_step(batch, batch_idx)
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # get training metrics
            _, metrics = run.training_get_artifacts_and_metrics()
            epoch_logs.update(metrics)

            # validation
            if (args.validation_force_interval is None) or (epoch == 0):
                force = False
            else:
                force = (epoch % args.validation_force_interval) == 0

            if (epoch >= (args.n_epochs * args.validation_skip)) or force:
                run.set_inference_mode()
                # we have multiple valid datasets due to multiple resolutions
                batch_idx = 0
                for i, valid_dataloader in enumerate(data.valid_dataloaders):
                    if isinstance(valid_dataloader.dataset,
                                  torch.utils.data.Subset):
                        # overfitting mode (dataset is wrapped using Subset)
                        camera = valid_dataloader.dataset.dataset.camera
                    else:
                        camera = valid_dataloader.dataset.camera
                    tqdm_desc = (f'Validation {i+1}/'
                                 f'{len(data.valid_dataloaders)} ({camera})')
                    for batch in tqdm(valid_dataloader,
                                      total=len(valid_dataloader),
                                      desc=tqdm_desc):
                        _ = run.validation_step(batch, batch_idx)
                        batch_idx += 1

                # get validation artifacts and metrics
                artifacts, examples, metrics = \
                    run.validation_get_artifacts_examples_metrics()
                epoch_logs.update(metrics)

                # checkpointing
                do_create_checkpoint = checkpoint_helper.check_for_checkpoint(
                    logs=epoch_logs,
                    add_checkpoint_metrics_to_logs=True
                )
                if epoch >= (args.n_epochs * args.checkpointing_skip) or force:
                    # we are allowed to store checkpoints
                    for ckpt_metric in do_create_checkpoint:
                        if not do_create_checkpoint[ckpt_metric]:
                            # no new best value, skip checkpointing
                            continue

                        # create new ckeckpoint
                        if args.checkpointing_best_only:
                            suffix = '_best'
                        else:
                            suffix = f'_epoch_{epoch:04d}'

                        mapped_name = \
                            checkpoint_helper.metric_mapping_joined[ckpt_metric]
                        ckpt_filepath = os.path.join(
                            checkpoints_path, f'ckpt_{mapped_name}{suffix}.pth')
                        # save checkpoint
                        if not args.enable_linear_probing:
                            ckpt = {
                                'state_dict': model.state_dict(),
                                'epoch': epoch,
                                'logs': epoch_logs
                            }
                            if scaler:
                                ckpt['scaler'] = scaler.state_dict()
                        else:
                            ckpt = {
                                'linear_probing_head_state_dict': model.linear_probing_head.state_dict(),
                                'epoch': epoch,
                                'logs': epoch_logs
                            }
                        torch.save(ckpt, ckpt_filepath)
                        print(f"Wrote checkpoint to: '{ckpt_filepath}'.")

                # store artifacts
                for key, value in artifacts.items():
                    fn = f'{key}__epoch_{epoch:04d}.npy'
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    np.save(os.path.join(artifacts_path, fn), value)

                # store / log examples
                wandb_examples = {}
                for key, value in examples.items():
                    fn = f'{key}__epoch_{epoch:04d}'
                    if isinstance(value, PIL.Image.Image):
                        value.save(os.path.join(examples_path, fn+'.png'),
                                   'PNG')
                        wandb_examples[key] = wandb.Image(value)
            else:
                wandb_examples = {}

            # update learning rate
            lr_scheduler.step()

            # resume checkpoint
            if ((epoch % args.resume_ckpt_interval) == 0 and epoch > 0) or \
                    (epoch == (args.n_epochs-1)):
                # save checkpoint containing state dict, optimizer, and lr
                # scheduler
                ckpt_filepath = os.path.join(checkpoints_path,
                                             'ckpt_resume.pth')

                if not args.enable_linear_probing:
                    ckpt = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'logs': epoch_logs
                    }
                    if scaler:
                        ckpt['scaler'] = scaler.state_dict()
                else:
                    ckpt = {
                        'linear_probing_head_state_dict': model.linear_probing_head.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'logs': epoch_logs
                    }
                    if scaler:
                        ckpt['scaler'] = scaler.state_dict()
                # write checkpoint file in paranoid mode
                torch.save(ckpt, ckpt_filepath+'.tmp')
                if os.path.isfile(ckpt_filepath):
                    # does exist only after first writing
                    os.remove(ckpt_filepath)
                os.rename(ckpt_filepath+'.tmp', ckpt_filepath)

                print(f"Wrote resume checkpoint to: '{ckpt_filepath}'.")

            # logging
            csv_logger.log(epoch_logs)
            wandb_logs = {**epoch_logs, **wandb_examples}
            wandb_logs = dict(sorted(wandb_logs.items()))
            wandb.log(wandb_logs, commit=True)
            if args.debug:
                print("Epoch logs:")
                pprint(epoch_logs)

    except Exception:
        # something went wrong -.-
        # store checkpoint
        ckpt_filepath = os.path.join(checkpoints_path,
                                     f'ckpt_error__epoch_{epoch:04d}.pth')
        if not args.enable_linear_probing:
            ckpt = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'logs': epoch_logs
            }
        else:
            ckpt = {
                'linear_probing_head_state_dict': model.linear_probing_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'logs': epoch_logs
            }
        print(f"Wrote checkpoint to: '{ckpt_filepath}'.")
        # log error
        log_filepath = os.path.join(results_path, 'error.log')
        with open(log_filepath, 'w') as f:
            traceback.print_exc(file=f)
        print(f"Wrote error log to: '{log_filepath}'.")

        # reraise error -> let the run crash
        raise

    # training done
    with open(os.path.join(results_path, 'finished'), 'w') as f:
        pass
    csv_logger.write()
    cprint_step("Done")


if __name__ == '__main__':
    main()
