#!/usr/bin/env python3
# modified from https://github.com/facebookresearch/SlowFast
"""Train a video classification model."""

import numpy as np
import pprint
import torch
import pdb
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from timm.utils import NativeScaler

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint_amp as cu # ...
import slowfast.utils.distributed as du # ...
import slowfast.utils.logging as logging # ...
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc # ...
import slowfast.visualization.tensorboard_vis as tb # ...
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter # ...
from slowfast.utils.multigrid import MultigridSchedule # ...

logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, loss_scaler, train_meter_v, train_meter_n, cur_epoch, cfg, writer=None):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        loss_scaler (scaler): scaler for loss.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter_v.iter_tic()
    train_meter_n.iter_tic()
    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

        mixup_fn_2 = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES_2
        )

    for cur_iter, (inputs, labels1, labels2, _, meta) in enumerate(train_loader):
        # print(11111111111111111111111)
        # print(labels1, labels2)
        # print(labels1.shape, labels2.shape)
        # assert False

        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels1 = labels1.cuda()
            labels2 = labels2.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter_v.data_toc()
        train_meter_n.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels1 = mixup_fn(inputs[0], labels1)
            samples, labels2 = mixup_fn_2(inputs[0], labels2)
            inputs[0] = samples

        with torch.cuda.amp.autocast():
            if cfg.DETECTION.ENABLE:
                x1, x2, unct1, unct2 = model(inputs, meta["boxes"]) 
            else:
                x1, x2, unct1, unct2 = model(inputs)
            # Explicitly declare reduction to mean.
            cls_loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            edl_loss_fun = losses.EvidenceLoss(cfg.MODEL.NUM_CLASSES)
            # edl_loss_fun_2 = losses.EvidenceLoss(cfg.MODEL.NUM_CLASSES_2)
            
            # Compute the loss.
            cls_loss_1 = cls_loss_fun(x1, labels1) 
            edl_loss_1 = edl_loss_fun(x1, labels1) 

            cls_loss_2 = cls_loss_fun(x2, labels2) 
            edl_loss_2 = edl_loss_fun(x2, labels2) 

            #print(cls_loss_1, edl_loss_1, cls_loss_2, edl_loss_2)
            loss = cls_loss_1 + edl_loss_1 + cls_loss_2 + edl_loss_2

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        # scaler => backward and step
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=cfg.SOLVER.CLIP_GRADIENT, parameters=model.parameters(), create_graph=is_second_order)
        
        if cfg.MIXUP.ENABLE:
            _top_max_k_vals_1, top_max_k_inds_1 = torch.topk(
                labels1, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels1.shape[0]), top_max_k_inds_1[:, 0]
            idx_top2 = torch.arange(labels1.shape[0]), top_max_k_inds_1[:, 1]
            x1[idx_top1] += x1[idx_top1]
            x1[idx_top2] = 0.0
            labels1 = top_max_k_inds_1[:, 0]

            _top_max_k_vals_2, top_max_k_inds_2 = torch.topk(
                labels2, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels2.shape[0]), top_max_k_inds_2[:, 0]
            idx_top2 = torch.arange(labels2.shape[0]), top_max_k_inds_2[:, 1]
            x2[idx_top1] += x2[idx_top1]
            x2[idx_top2] = 0.0
            labels2 = top_max_k_inds_2[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter_v.update_stats(None, None, None, loss, lr)
            train_meter_n.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )
        else:
            top1_err_v, top5_err_v, top1_err_n, top5_err_n = None, None, None, None
            
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(x1, labels1, (1, 5))
                top1_err_v, top5_err_v = [
                    (1.0 - x / x1.size(0)) * 100.0 for x in num_topks_correct
                ]
                
                num_topks_correct = metrics.topks_correct(x2, labels2, (1, 5))
                top1_err_n, top5_err_n = [
                    (1.0 - x / x2.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err_v, top5_err_v, top1_err_n, top5_err_n = du.all_reduce(
                        [loss, top1_err_v, top5_err_v, top1_err_n, top5_err_n]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err_v, top5_err_v, top1_err_n, top5_err_n = (
                    loss.item(),
                    top1_err_v.item(),
                    top5_err_v.item(),
                    top1_err_n.item(),
                    top5_err_n.item(),
                )

            # Update and log stats.
            train_meter_v.update_stats(
                top1_err_v,
                top5_err_v,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            train_meter_n.update_stats(
                top1_err_n,
                top5_err_n,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err_v": top1_err_v,
                        "Train/Top5_err_v": top5_err_v,
                        "Train/Top1_err_n": top1_err_n,
                        "Train/Top5_err_n": top5_err_n,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter_v.iter_toc()  # measure allreduce for this meter
        train_meter_v.log_iter_stats(cur_epoch, cur_iter)
        train_meter_v.iter_tic()
        train_meter_n.iter_toc()  # measure allreduce for this meter
        train_meter_n.log_iter_stats(cur_epoch, cur_iter)
        train_meter_n.iter_tic()

    # Log epoch stats.
    train_meter_v.log_epoch_stats(cur_epoch)
    train_meter_v.reset()
    train_meter_n.log_epoch_stats(cur_epoch)
    train_meter_n.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter_v, val_meter_n, loss_scaler, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        loss_scaler (scaler): scaler for loss.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter_v.iter_tic()
    val_meter_n.iter_tic()

    for cur_iter, (inputs, labels1, labels2, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels1 = labels1.cuda()
            labels2 = labels2.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter_v.data_toc()
        val_meter_n.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            x1, x2, unct1, unct2 = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                x1 = x1.cpu()
                x2 =x2.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                x1 = torch.cat(du.all_gather_unaligned(x1), dim=0)
                x2 = torch.cat(du.all_gather_unaligned(x2), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter_v.iter_toc()
            val_meter_n.iter_toc()
            # Update and log stats.
            val_meter_v.update_stats(x1, ori_boxes, metadata)
            val_meter_n.update_stats(x2, ori_boxes, metadata)

        else:
            x1, x2, unct1, unct2 = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    x1, labels1 = du.all_gather([x1, labels1])
                    x2, labels2 = du.all_gather([x2, labels2])
            else:
                # Compute the errors.
                num_topks_correct_1 = metrics.topks_correct(x1, labels1, (1, 5))
                num_topks_correct_2 = metrics.topks_correct(x2, labels2, (1, 5))

                # Combine the errors across the GPUs.
                top1_err_v, top5_err_v = [
                    (1.0 - x / x1.size(0)) * 100.0 for x in num_topks_correct_1
                ]
                top1_err_n, top5_err_n = [
                    (1.0 - x / x2.size(0)) * 100.0 for x in num_topks_correct_2
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err_v, top5_err_v = du.all_reduce([top1_err_v, top5_err_v])
                    top1_err_n, top5_err_n = du.all_reduce([top1_err_n, top5_err_n])

                # Copy the errors from GPU to CPU (sync point).
                top1_err_v, top5_err_v, top1_err_n, top5_err_n = top1_err_v.item(), top5_err_v.item(), top1_err_n.item(), top5_err_n.item()

                val_meter_v.iter_toc()
                val_meter_n.iter_toc()
                # Update and log stats.
                val_meter_v.update_stats(
                    top1_err_v,
                    top5_err_v,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                val_meter_n.update_stats(
                    top1_err_n,
                    top5_err_n,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err_v": top1_err_v, "Val/Top5_err_v": top5_err_v, "Val/Top1_err_n": top1_err_n, "Val/Top5_err_n": top5_err_n},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter_v.update_predictions(x1, labels1)
            val_meter_n.update_predictions(x2, labels2)

        val_meter_v.log_iter_stats(cur_epoch, cur_iter)
        val_meter_v.iter_tic()
        val_meter_n.log_iter_stats(cur_epoch, cur_iter)
        val_meter_n.iter_tic()

    # Log epoch stats.
    val_meter_v.log_epoch_stats(cur_epoch)
    val_meter_n.log_epoch_stats(cur_epoch)

    # write to tensorboard format if available.
    writer = None
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter_v.full_map}, global_step=cur_epoch
            )
        else:
            all_preds_1 = [pred.clone().detach() for pred in val_meter_v.all_preds]
            all_preds_2 = [pred.clone().detach() for pred in val_meter_n.all_preds]
            all_labels1 = [
                label1.clone().detach() for label1 in val_meter_v.all_labels
            ]
            all_labels2 = [
                label2.clone().detach() for label2 in val_meter_n.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds_1 = [pred.cpu() for pred in all_preds_1]
                all_preds_2 = [pred.cpu() for pred in all_preds_2]
                all_labels1 = [label1.cpu() for label1 in all_labels1]
                all_labels2 = [label2.cpu() for label2 in all_labels2]
            writer.plot_eval(
                x1 = all_preds_1, labels1=all_labels1, global_step=cur_epoch
            )
            writer.plot_eval(
                x2 = all_preds_2, labels2=all_labels2, global_step=cur_epoch
            )

    val_meter_v.reset()
    val_meter_n.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Loss scaler
    loss_scaler = NativeScaler()

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter_v = TrainMeter(len(train_loader), cfg)
    train_meter_n = TrainMeter(len(train_loader), cfg)
    val_meter_v = ValMeter(len(val_loader), cfg)
    val_meter_n = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        loss_scaler,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter_v,
        train_meter_n,
        val_meter_v,
        val_meter_n,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
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

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Loss scaler
    loss_scaler = NativeScaler()

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, loss_scaler)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter_v = AVAMeter(len(train_loader), cfg, mode="train")
        train_meter_n = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter_v = AVAMeter(len(val_loader), cfg, mode="val")
        val_meter_n = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter_v = TrainMeter(len(train_loader), cfg)
        train_meter_n = TrainMeter(len(train_loader), cfg)
        val_meter_v = ValMeter(len(val_loader), cfg)
        val_meter_n = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    loss_scaler,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter_v,
                    train_meter_n,
                    val_meter_v,
                    val_meter_n,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        # eval_epoch(val_loader, model, val_meter_v, val_meter_n, loss_scaler, cur_epoch, cfg, writer) # test20231016
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader, model, optimizer, loss_scaler, train_meter_v, train_meter_n, cur_epoch, cfg, writer
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, loss_scaler, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter_v, val_meter_n, loss_scaler, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()
