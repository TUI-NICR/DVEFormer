# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import os

import pytest
import torch

from nicr_mt_scene_analysis.data.preprocessing.base import APPLIED_PREPROCESSING_KEY
from nicr_mt_scene_analysis.data.preprocessing.resize import Resize
from nicr_mt_scene_analysis.testing.onnx import export_onnx_model

from dveformer.args import ArgParserDVEFormer
from dveformer.data import get_dataset
from dveformer.model import DVEFormer


def model_test(tasks,
               modalities,
               backbone,
               do_postprocessing,
               training,
               tmp_path,
               additional_args=None,
               enable_linear_probing=False):

    parser = ArgParserDVEFormer()
    cli_args = [
        '--input-modalities', *modalities,
        '--tasks', *tasks,
        '--rgbd-encoder-backbone', backbone,
        '--rgb-encoder-backbone', backbone,
        '--depth-encoder-backbone', backbone,
        '--no-pretrained-backbone',
        '--dataset', 'nyuv2',
        # Only test low res model because fullres gives oom in CI
        '--dense-visual-embedding-decoder-n-upsamplings', '0',
        # Use small batch size to avoid OOM in CI
        '--batch-size', '2',
        '--validation-batch-size', '2',
    ]
    if enable_linear_probing:
        cli_args.append('--enable-linear-probing')
    if additional_args:
        cli_args.extend(additional_args)

    args = parser.parse_args(cli_args, verbose=False)

    dataset = get_dataset(args, split='train')
    dataset_config = dataset.config

    # create model
    model = DVEFormer(args, dataset_config=dataset_config)
    if not training:
        model.eval()

    # determine input
    batch_size = 3
    input_shape = (480, 640)
    batch = {}
    if 'rgb' in args.input_modalities or 'rgbd' in args.input_modalities:
        batch['rgb'] = torch.randn((batch_size, 3)+input_shape)
    if 'depth' in args.input_modalities or 'rgbd' in args.input_modalities:
        batch['depth'] = torch.randn((batch_size, 1)+input_shape)


    # Add applied preprocessing to batch which is required for postprocessing
    batch[APPLIED_PREPROCESSING_KEY] = [
        [{
            'type': Resize.__name__,
            'valid_region_slice_y': slice(0, input_shape[0]),
            'valid_region_slice_x': slice(0, input_shape[1]),
        },]
    ]*batch_size

    if not training and do_postprocessing:
        # for inference postprocessing, inputs in full resolution are required
        if 'rgb' in batch:
            batch['rgb_fullres'] = batch['rgb'].clone()
        if 'depth' in batch:
            batch['depth_fullres'] = batch['depth'].clone()

    # apply model
    outputs = model(batch, do_postprocessing=do_postprocessing)

    # some simple checks for output
    if do_postprocessing:
        assert isinstance(outputs, dict)
    else:
        assert isinstance(outputs, list)
    assert outputs

    # export model to ONNX
    if not training and do_postprocessing:
        # stop here: inference postprocessing is challenging (no onnx export)
        return
    # determine filename and filepath
    tasks_str = '+'.join(tasks)
    modalities_str = '+'.join(modalities)
    filename = f'model_{modalities_str}_{tasks_str}'
    filename += f'__backbone_{backbone}'
    filename += f'__train{training}'
    filename += f'__post_{do_postprocessing}'
    filename += '.onnx'
    filepath = os.path.join(tmp_path, filename)
    # export
    # note, the last element in input tuple is interpreted as named args
    # if no named args should be passed use
    x = (batch, {'do_postprocessing': do_postprocessing})
    export_onnx_model(filepath, model, x)


@pytest.mark.parametrize('modalities', (('rgb',),
                                        ('depth',),
                                        ('rgbd',)))
@pytest.mark.parametrize('backbone', ('swin-t', 'swin-t-v2',
                                      'swin-t-128', 'swin-t-v2-128'))
@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
def test_dense_visual_embedding_model(modalities, backbone, do_postprocessing,
                                      training, tmp_path):
    """Test DVEFormer dense visual embedding task for full and low resolution"""
    model_test(
        tasks=('dense-visual-embedding',),
        modalities=modalities,
        backbone=backbone,
        do_postprocessing=do_postprocessing,
        training=training,
        tmp_path=tmp_path
    )


@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
def test_model_less_downsampling_skips(do_postprocessing, training, tmp_path):
    """Test DVEFormer dense visual embedding with adjusted skip configuration"""
    model_test(
        tasks=('dense-visual-embedding',),
        modalities=('rgbd',),
        backbone='swin-multi-t-v2-128',
        do_postprocessing=do_postprocessing,
        training=training,
        tmp_path=tmp_path,
        additional_args=[
            '--encoder-decoder-skip-downsamplings', '4', '8',
            '--dense-visual-embedding-decoder-n-channels', '256', '128', '64',
        ]
    )


@pytest.mark.parametrize('do_postprocessing', (False, True))
@pytest.mark.parametrize('training', (False, True))
def test_model_linear_probing(do_postprocessing, training, tmp_path):
    """Test DVEFormer dense visual embedding with linear probing enabled"""
    model_test(
        tasks=('dense-visual-embedding',),
        modalities=('rgbd',),
        backbone='swin-t',
        do_postprocessing=do_postprocessing,
        training=training,
        tmp_path=tmp_path,
        enable_linear_probing=True,
    )
