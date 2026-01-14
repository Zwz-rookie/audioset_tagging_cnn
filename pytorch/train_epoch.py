import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from utilities import (create_folder, get_filename, create_logging, Mixup, 
    StatisticsContainer)
from models import (Cnn14, Cnn14_no_specaug, Cnn14_no_dropout, 
    Cnn6, Cnn10, ResNet22, ResNet38, ResNet54, Cnn14_emb512, Cnn14_emb128, 
    Cnn14_emb32, MobileNetV1, MobileNetV2, MobileNetV2_Mod, LeeNet11, LeeNet24, DaiNet19,
    Res1dNet31, Res1dNet51, Wavegram_Cnn14, Wavegram_Logmel_Cnn14, 
    Wavegram_Logmel128_Cnn14, Cnn14_16k, Cnn14_16k_Mod, Cnn14_8k, Cnn14_mel32, Cnn14_mel128,
    Cnn14_mixup_time_domain, Cnn14_DecisionLevelMax, Cnn14_DecisionLevelAtt)
from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
    do_mixup)
from data_generator_my import (AudioSetDataset, TrainSampler, BalancedTrainSampler,
    AlternateTrainSampler, EvaluateSampler, collate_fn)
from evaluate import Evaluator
import config
from losses import get_loss_func

checkpoint_path = "MobileNetV2_Mod_GM_V11.pth"

def train(args):
    """Train AudioSet tagging model with epoch-based approach. 
    
    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_epoch: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """
    global checkpoint_path
    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_epoch = args.resume_epoch
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    num_workers = 8
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)

    # Paths
    black_list_csv = None
    
    train_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes', 
        '{}.h5'.format(data_type))

    eval_bal_indexes_hdf5_path = os.path.join(workspace, 
        'hdf5s', 'indexes', 'balanced_train.h5')

    eval_test_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes', 
        'eval.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'sample_rate={},mel_bins={}'.format(sample_rate, mel_bins),
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    
    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'sample_rate={},mel_bins={}'.format(sample_rate, mel_bins),
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        'sample_rate={},mel_bins={}'.format(sample_rate, mel_bins),
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'
    
    # Model
    print("checkpoint_path路径： ", checkpoint_path)
    Model = eval(model_type)
    if not model_type.endswith("_Mod"):
        model = Model(sample_rate=sample_rate, window_size=window_size,
            hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
            classes_num=classes_num)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print("✅ 成功加载 Cnn14_16k 模型！")

    elif model_type == 'MobileNetV2_Mod':
        if "Mod" in checkpoint_path:
            model = Model(mel_bins=mel_bins, classes_num=classes_num)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print("✅ 成功加载 MobileNetV2_Mod 模型！")
        else:
            # 2. 初始化新模型
            model = Model(mel_bins=mel_bins, classes_num=classes_num)
            model_state = model.state_dict()
            # 1. 加载预训练模型权重
            pretrained = torch.load(checkpoint_path, map_location=device)
            pretrained_state = pretrained["model"] if "model" in pretrained else pretrained
            # 3. 过滤掉前端特征提取层，只保留 bn0 及之后的
            filtered_state = {
                k: v for k, v in pretrained_state.items()
                if not (k.startswith("feature_extractor") or k.startswith("fc_audioset")
                        or k.startswith("spectrogram_extractor") or k.startswith("logmel_extractor"))
            }
            # 4. 更新 state_dict
            model_state.update(filtered_state)
            # 5. 加载
            model.load_state_dict(model_state)
            print("✅ 成功加载 MobileNetV2 bn0 及后续层参数！")

    else:
        if "Mod" in checkpoint_path:
            model = Model(mel_bins=mel_bins, classes_num=classes_num)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print("✅ 成功加载 Cnn14_16k_Mod 模型！")
        else:
            # 1. 加载预训练模型权重
            pretrained = torch.load(checkpoint_path, map_location=device)
            pretrained_state = pretrained["model"] if "model" in pretrained else pretrained

            # 2. 初始化新模型
            model = Model(mel_bins=mel_bins, classes_num=classes_num)
            model_state = model.state_dict()

            # 3. 过滤掉前端特征提取层，只保留 bn0 及之后的
            filtered_state = {
                k: v for k, v in pretrained_state.items()
                if k.startswith("bn0") or k.startswith("conv_block") or k.startswith("fc1")
            }

            # 4. 更新 state_dict
            model_state.update(filtered_state)

            # 5. 加载
            model.load_state_dict(model_state)

            print("✅ 成功加载 bn0 及后续层参数！")

    params_num = count_parameters(model)
    logging.info('Parameters num: {}'.format(params_num))
    
    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    dataset = AudioSetDataset(sample_rate=sample_rate)

    # Train sampler
    if balanced == 'none':
        Sampler = TrainSampler
    elif balanced == 'balanced':
        Sampler = BalancedTrainSampler
    elif balanced == 'alternate':
        Sampler = AlternateTrainSampler
    
    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
        black_list_csv=black_list_csv)
    
    # 打印训练数据集长度
    print(f"训练数据集长度: {train_sampler.audios_num} 个样本")
    
    # Evaluate sampler
    eval_bal_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=batch_size)

    eval_test_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)

    # 打印验证和测试数据集长度
    print(f"验证集(balanced)长度: {eval_bal_sampler.audios_num} 个样本")
    print(f"测试集长度: {eval_test_sampler.audios_num} 个样本")

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_sampler=eval_bal_sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True)

    # eval_test_loader = torch.utils.data.DataLoader(dataset=dataset,
    #     batch_sampler=eval_test_sampler, collate_fn=collate_fn,
    #     num_workers=num_workers, pin_memory=True)

    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    # Evaluator
    evaluator = Evaluator(model=model)
        
    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Scheduler (optional, but recommended for better training)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Variables for tracking best model
    best_train_loss = float('inf')
    early_stop_count = 0

    train_bgn_time = time.time()
    
    # Resume training
    if resume_epoch > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            '{}_epochs.pth'.format(resume_epoch))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_epoch * len(train_loader))
        current_epoch = checkpoint['epoch']
        if 'best_train_loss' in checkpoint:
            best_train_loss = checkpoint['best_train_loss']
        else:
            best_train_loss = float('inf')
        early_stop_count = checkpoint['early_stop_count']

    else:
        current_epoch = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    # Training loop
    for epoch in range(current_epoch, early_stop):
        print(f"--- Epoch {epoch + 1}/{early_stop} ---")
        
        # Training phase
        model.train()
        train_loss = 0.0
        total_batches = 0
        time1 = time.time()
        
        for batch_data_dict in train_loader:
            # Mixup lambda
            if 'mixup' in augmentation:
                batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                    batch_size=len(batch_data_dict['waveform']))

            # Move data to device
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
            
            # Forward
            if 'mixup' in augmentation:
                batch_output_dict = model(batch_data_dict['waveform'], 
                    batch_data_dict['mixup_lambda'])
                batch_target_dict = {'target': do_mixup(batch_data_dict['target'], 
                    batch_data_dict['mixup_lambda'])}
            else:
                batch_output_dict = model(batch_data_dict['waveform'], None)
                batch_target_dict = {'target': batch_data_dict['target']}

            # Loss
            loss = loss_func(batch_output_dict, batch_target_dict)
            train_loss += loss.item()
            total_batches += 1

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if total_batches % 100 == 0:
                print('--- Epoch: {}, Batch: {}, Loss: {:.6f}, Time: {:.3f} s ---'\
                    .format(epoch + 1, total_batches, loss.item(), time.time() - time1))
                time1 = time.time()
        
        avg_train_loss = train_loss / total_batches
        logging.info('Epoch: {}, Average training loss: {:.6f}'.format(epoch + 1, avg_train_loss))
        
        # Evaluation phase
        model.eval()
        train_fin_time = time.time()

        # bal_statistics = evaluator.evaluate(eval_bal_loader)
        # test_statistics = evaluator.evaluate(eval_test_loader)

        # val_map = np.mean(bal_statistics['average_precision'])
        # test_map = np.mean(test_statistics['average_precision'])

        # logging.info('Validate bal mAP: {:.3f}'.format(val_map))
        # logging.info('Validate test mAP: {:.3f}'.format(test_map))
        #
        # statistics_container.append(epoch * len(train_loader), bal_statistics, data_type='bal')
        # statistics_container.append(epoch * len(train_loader), test_statistics, data_type='test')
        # statistics_container.dump()

        train_time = train_fin_time - train_bgn_time
        validate_time = time.time() - train_fin_time

        logging.info(
            'Epoch: {}, train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(epoch + 1, train_time, validate_time))

        logging.info('------------------------------------')

        # Update learning rate scheduler
        scheduler.step(avg_train_loss)

        # Save best model based on training loss
        if epoch == 0 or avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            early_stop_count = 0
            checkpoint = {
                'iteration': epoch + 1,
                'model': model.module.state_dict(),
                'sampler': train_sampler.state_dict()}
            # checkpoint = {
            #     'epoch': epoch + 1,
            #     'model': model.module.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'scheduler': scheduler.state_dict(),
            #     'sampler': train_sampler.state_dict(),
            #     'best_train_loss': best_train_loss,
            #     'early_stop_count': early_stop_count}

            checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
            
            torch.save(checkpoint, checkpoint_path)
            logging.info('Best model saved to {}'.format(checkpoint_path))
        else:
            early_stop_count += 1
            logging.info('Early stop count: {}/{}'.format(early_stop_count, args.patience))
            
            # Early stop if no improvement for 'patience' epochs
            if early_stop_count >= args.patience:
                logging.info('Early stopping triggered after {} epochs without improvement'.format(early_stop_count))
                break
        
        # Save checkpoint for current epoch
        # checkpoint = {
        #     'epoch': epoch + 1,
        #     'model': model.module.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'scheduler': scheduler.state_dict(),
        #     'sampler': train_sampler.state_dict(),
        #     'best_train_loss': best_train_loss,
        #     'early_stop_count': early_stop_count}
        checkpoint = {
            'iteration': epoch + 1,
            'model': model.module.state_dict(),
            'sampler': train_sampler.state_dict()}

        checkpoint_path = os.path.join(
            checkpoints_dir, '{}_epochs.pth'.format(epoch + 1))
        
        torch.save(checkpoint, checkpoint_path)
        logging.info('Epoch checkpoint saved to {}'.format(checkpoint_path))
        
        train_bgn_time = time.time()
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, default='full_train', choices=['balanced_train', 'full_train'])
    parser_train.add_argument('--sample_rate', type=int, default=16000)
    parser_train.add_argument('--window_size', type=int, default=1024)
    parser_train.add_argument('--hop_size', type=int, default=320)
    parser_train.add_argument('--mel_bins', type=int, default=64)
    parser_train.add_argument('--fmin', type=int, default=50)
    parser_train.add_argument('--fmax', type=int, default=14000) 
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, default='clip_bce', choices=['clip_bce'])
    parser_train.add_argument('--balanced', type=str, default='balanced', choices=['none', 'balanced', 'alternate'])
    parser_train.add_argument('--augmentation', type=str, default='mixup', choices=['none', 'mixup'])
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--resume_epoch', type=int, default=0)
    parser_train.add_argument('--resume_iteration', type=int, default=0)
    parser_train.add_argument('--early_stop', type=int, default=100)
    parser_train.add_argument('--patience', type=int, default=10)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')
