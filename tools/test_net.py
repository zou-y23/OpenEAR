#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from iopath.common.file_io import g_pathmgr

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter_v, test_meter_n, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter_v.iter_tic()
    test_meter_n.iter_tic()
    acc_topk1_all = 0.
    acc_topk5_all = 0.
    for cur_iter, (inputs, labels1, labels2, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels1 = labels1.cuda()
            labels2 = labels2.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter_v.data_toc()
        test_meter_n.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            x1, x2, unct1, unct2 = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            x1 = x1.detach().cpu() if cfg.NUM_GPUS else x1.detach()
            x2 = x2.detach().cpu() if cfg.NUM_GPUS else x2.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                x1 = torch.cat(du.all_gather_unaligned(x1), dim=0)
                x2 = torch.cat(du.all_gather_unaligned(x2), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter_v.iter_toc()
            test_meter_n.iter_toc()
            # Update and log stats.
            test_meter_v.update_stats(x1, ori_boxes, metadata)
            test_meter_v.log_iter_stats(None, cur_iter)
            test_meter_n.update_stats(x2, ori_boxes, metadata)
            test_meter_n.log_iter_stats(None, cur_iter)

        else:
            # Perform the forward pass.
            x1, x2, unct1, unct2 = model(inputs)
            
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                x1, labels1, video_idx = du.all_gather(
                    [x1, labels1, video_idx]
                )
                x2, labels2, video_idx = du.all_gather(
                    [x2, labels2, video_idx]
                )
            if cfg.NUM_GPUS:
                x1 = x1.cpu()
                x2 = x2.cpu()
                labels1 = labels1.cpu()
                labels2 = labels2.cpu()
                video_idx = video_idx.cpu()
            '''
            print(unct1.shape)
            print(unct1)
            _, new_x1 = torch.max(x1, dim=1)
            print(new_x1.shape)
            print(new_x1)
            print(labels1.shape)
            print(labels1)
            print(unct2.shape)
            print(unct2)
            _, new_x2 = torch.max(x2, dim=1)
            print(new_x2.shape)
            print(new_x2)
            print(labels2.shape)
            print(labels2)
            assert False
            '''

            test_meter_v.iter_toc()
            test_meter_n.iter_toc()
            # Update and log stats.
            test_meter_v.update_stats(
                x1.detach(), labels1.detach(), video_idx.detach()
            )
            test_meter_v.log_iter_stats(cur_iter)
            test_meter_n.update_stats(
                x2.detach(), labels2.detach(), video_idx.detach()
            )
            test_meter_n.log_iter_stats(cur_iter)
        test_meter_v.iter_tic()
        test_meter_n.iter_tic()
        for j in range(len(labels1)):
            _, new_x1_top1 = torch.topk(x1[j], k=1)
            _, new_x1_top5 = torch.topk(x1[j], k=5)
            _, new_x2_top1 = torch.topk(x2[j], k=1)
            _, new_x2_top5 = torch.topk(x2[j], k=5)
            new_label1 = labels1[j]
            new_label2 = labels2[j]
            if torch.any(torch.eq(new_x1_top1, new_label1)) and torch.any(torch.eq(new_x2_top1, new_label2)):
                acc_topk1_all = acc_topk1_all + 1.
            if torch.any(torch.eq(new_x1_top5, new_label1)) and torch.any(torch.eq(new_x2_top5, new_label2)):
                acc_topk5_all = acc_topk5_all + 1.
    acc_topk1 = acc_topk1_all / test_loader.dataset.num_videos
    acc_topk5 = acc_topk5_all / test_loader.dataset.num_videos
    print('acc_all_topk1:', acc_topk1, 'acc_all_topk5:', acc_topk5)

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds_1 = test_meter_v.video_preds.clone().detach()
        all_labels1 = test_meter_v.video_labels
        all_preds_2 = test_meter_n.video_preds.clone().detach()
        all_labels2 = test_meter_n.video_labels
        if cfg.NUM_GPUS:
            all_preds_1 = all_preds_1.cpu()
            all_labels1 = all_labels1.cpu()
            all_preds_2 = all_preds_2.cpu()
            all_labels2 = all_labels2.cpu()

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with g_pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds_1, all_labels1], f)
                    pickle.dump([all_preds_2, all_labels2], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter_v.finalize_metrics()
    test_meter_n.finalize_metrics()
    return test_meter_v, test_meter_n


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter_v = AVAMeter(len(test_loader), cfg, mode="test")
        test_meter_n = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter_v = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )
        test_meter_n = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES_2,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter_v, test_meter_n = perform_test(test_loader, model, test_meter_v, test_meter_n, cfg, writer)
    if writer is not None:
        writer.close()

    
    file_name = f'{cfg.DATA.NUM_FRAMES}x{cfg.DATA.TEST_CROP_SIZE}x{cfg.TEST.NUM_ENSEMBLE_VIEWS}x{cfg.TEST.NUM_SPATIAL_CROPS}.pkl'
    with g_pathmgr.open(os.path.join(
        cfg.OUTPUT_DIR, file_name),
        'wb'
    ) as f:
        result = {
            'video_preds_v': test_meter_v.video_preds.cpu().numpy(),
            'video_labels_v': test_meter_v.video_labels.cpu().numpy(),
            'video_preds_n': test_meter_n.video_preds.cpu().numpy(),
            'video_labels_n': test_meter_n.video_labels.cpu().numpy()
        }
        pickle.dump(result, f)