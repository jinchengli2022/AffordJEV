"""Fixed-step training and RGB-only inference for the local OOAL UMD baseline."""
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...common import read, read_jsonl, save_mask, sha, write, write_jsonl
from .data import OneShot, load_rgb


def initialize(config):
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires CUDA')


def model_for(config):
    root = Path(config['ooal_root']).resolve(strict=True)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != config['ooal_commit']:
        raise ValueError(f'Unexpected OOAL source version: {commit}')
    for key in ['dino_repository', 'dino_weights', 'clip_weights']:
        Path(config[key]).resolve(strict=True)
    cached_dino = Path(torch.hub.get_dir()) / 'checkpoints' / 'dinov2_vitb14_pretrain.pth'
    if cached_dino.resolve() != Path(config['dino_weights']).resolve():
        raise ValueError('DINO torch.hub cache differs from the configured pretrained file')
    sys.path.insert(0, str(root))
    from models.ooal import Net
    network = Net(SimpleNamespace(class_names=config['class_names']), 768, 512,
                  dino_pretrained='dinov2_vitb14', dino_repository=config['dino_repository'],
                  clip_checkpoint=config['clip_weights']).cuda()
    if any(p.requires_grad for p in network.dino_model.parameters()):
        raise RuntimeError('DINO backbone must be frozen')
    if any(p.requires_grad for p in network.aff_text_encoder.parameters()):
        raise RuntimeError('CLIP text encoder must be frozen')
    return network


def source_record(config):
    root = Path(config['ooal_root'])
    patch = subprocess.check_output(['git', '-C', str(root), 'diff', '--', 'models/ooal.py'], text=True)
    sources = list((root / 'models').rglob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    return {
        'ooal_commit': config['ooal_commit'], 'ooal_patch': patch,
        'sources_sha256': {str(p.resolve()): sha(p) for p in sources},
        'foundation_weights': {key: {'path': config[key], 'sha256': sha(config[key])}
                               for key in ['dino_weights', 'clip_weights']},
        'packages': {name: importlib.metadata.version(name)
                     for name in ['torch', 'torchvision', 'numpy', 'scipy', 'Pillow']},
        'python': sys.version, 'conda_prefix': sys.prefix,
        'gpu': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }


def head_state(network):
    return {name: value.detach().cpu() for name, value in network.state_dict().items()
            if not name.startswith(('dino_model.', 'aff_text_encoder.'))}


def save_checkpoint(network, optimizer, scheduler, step, config, train_sha, output):
    target = Path(output) / f'iteration_{step:06d}.pth'
    temporary = target.with_suffix('.tmp')
    torch.save({'iteration': step, 'model_state_dict': head_state(network),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'config': config, 'train_manifest_sha256': train_sha,
                'task_checkpoint_initialization': 'random; no AGD20K affordance checkpoint'}, temporary)
    temporary.replace(target)
    return target


def train(config, output, checkpoints, steps=None):
    initialize(config)
    out, weights = Path(output).resolve(), Path(checkpoints).resolve()
    for folder in [out, weights]:
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f'Training output must be empty: {folder}')
        folder.mkdir(parents=True, exist_ok=True)
    iterations = config['iterations'] if steps is None else steps
    if iterations <= 0:
        raise ValueError('iterations must be positive')
    manifest = Path(config['train_manifest']).resolve(strict=True)
    meta = read(manifest.with_suffix('.meta.json'))
    train_sha = sha(manifest)
    if train_sha != meta['train_manifest_sha256'] or meta['instance_overlap'] or meta['category_overlap']:
        raise ValueError('Training manifest audit failed')
    write(out / 'config.json', config)
    write(out / 'split_audit.json', meta)
    write(out / 'source_and_environment.json', source_record(config))
    write_jsonl(out / 'training_manifest.jsonl', read_jsonl(manifest))
    dataset = OneShot(manifest, config['resize_size'], config['input_size'], config['horizontal_flip_probability'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['workers'], pin_memory=True,
                        persistent_workers=config['workers'] > 0,
                        generator=torch.Generator().manual_seed(config['seed']))
    network = model_for(config)
    network.train()
    network.dino_model.eval()
    network.aff_text_encoder.eval()
    parameters = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=config['learning_rate'], momentum=config['momentum'],
                                weight_decay=config['weight_decay'], nesterov=config['nesterov'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    write(out / 'parameters.json', {
        'trainable_parameters': sum(p.numel() for p in parameters),
        'total_parameters': sum(p.numel() for p in network.parameters()),
        'trainable_names': [name for name, p in network.named_parameters() if p.requires_grad],
        'frozen_visual_encoder': True, 'frozen_text_encoder': True,
    })
    state = {'status': 'running', 'pid': os.getpid(), 'iteration': 0, 'total_iterations': iterations,
             'training_samples': len(dataset), 'seed': config['seed'], 'test_evaluations_during_training': 0,
             'development_only': steps is not None, 'checkpoint_selection': config['selection']}
    write(out / 'status.json', state)
    write(weights / 'status.json', state)
    log = (out / 'loss.jsonl').open('w', buffering=1)
    started = time.perf_counter()
    iterator = iter(loader)
    losses, grad_check = [], None
    latest = None
    step = 0
    try:
        for step in tqdm(range(1, iterations + 1), desc='OOAL UMD train', mininterval=5):
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, labels = next(iterator)
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = network(images, return_logits=True)
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            if step == 1:
                grad_check = {
                    'prompt_gradient_l1': float(network.prompt_learner.ctx.grad.abs().sum()),
                    'decoder_gradient_l1': float(network.seg_decoder.decoder.layers[0].linear1.weight.grad.abs().sum()),
                    'frozen_parameters_with_grad': [name for name, p in network.named_parameters()
                                                    if not p.requires_grad and p.grad is not None],
                    'input_shape': list(images.shape), 'label_ids': labels.unique().tolist(),
                    'output_shape': list(logits.shape),
                }
                if grad_check['frozen_parameters_with_grad'] or min(grad_check['prompt_gradient_l1'], grad_check['decoder_gradient_l1']) <= 0:
                    raise RuntimeError(f'Gradient audit failed: {grad_check}')
                write(out / 'first_step_audit.json', grad_check)
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.detach())
            losses.append(loss_value)
            if step % config['log_interval'] == 0 or step in {1, iterations}:
                elapsed = time.perf_counter() - started
                event = {'iteration': step, 'loss': loss_value, 'recent_mean_loss': float(np.mean(losses[-100:])),
                         'learning_rate': scheduler.get_last_lr()[0], 'elapsed_seconds': elapsed,
                         'eta_seconds': elapsed / step * (iterations - step)}
                log.write(json.dumps(event) + '\n')
                state.update(event)
                write(out / 'status.json', state)
                print(f"STEP {step}/{iterations} CE={event['recent_mean_loss']:.6f} ETA={event['eta_seconds']/60:.1f} min", flush=True)
            if step % config['checkpoint_interval'] == 0 or step == iterations:
                latest = save_checkpoint(network, optimizer, scheduler, step, config, train_sha, weights)
                state['latest_checkpoint'] = str(latest)
                write(out / 'status.json', state)
                write(weights / 'status.json', state)
        state.update(status='complete', iteration=step, elapsed_seconds=time.perf_counter() - started,
                     selected_checkpoint=str(latest), last_loss=losses[-1], recent_mean_loss=float(np.mean(losses[-100:])))
    except BaseException as exc:
        state.update(status='failed', iteration=step, error=f'{type(exc).__name__}: {exc}',
                     elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        log.close()
        write(out / 'status.json', state)
        write(weights / 'status.json', state)
    return state


def infer(config, checkpoint, output, manifest=None, limit=0):
    initialize(config)
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Inference output must be empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    manifest = config['test_manifest'] if manifest is None else manifest
    rows = read_jsonl(manifest)
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError('Empty inference manifest')
    network = model_for(config)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['config'] != config or saved['train_manifest_sha256'] != sha(config['train_manifest']):
        raise ValueError('Checkpoint training configuration or data does not match')
    incompatible = network.load_state_dict(saved['model_state_dict'], strict=False)
    if incompatible.unexpected_keys or any(not k.startswith(('dino_model.', 'aff_text_encoder.')) for k in incompatible.missing_keys):
        raise RuntimeError(f'Invalid head checkpoint: {incompatible}')
    del saved
    network.eval()
    write(out / 'config_snapshot.json', config)
    write(out / 'affordances_snapshot.json', read(config['definitions']))
    write_jsonl(out / 'evaluation_manifest.jsonl', rows)
    manifest_sha = sha(out / 'evaluation_manifest.jsonl')
    state = {'status': 'running', 'model': config['name'], 'samples': len(rows), 'pid': os.getpid(),
             'evaluation_manifest_sha256': manifest_sha, 'checkpoint': str(Path(checkpoint).resolve()),
             'checkpoint_sha256': sha(checkpoint), 'reference_annotations_used_in_inference': False,
             'whole_object_prohibition': False, 'limited_run': bool(limit), 'input_size': config['input_size']}
    write(out / 'run.json', state)
    # Warm up with a training RGB. Ground-truth arrays are never read here.
    warm_path = read_jsonl(config['train_manifest'])[0]['rgb']
    warm = load_rgb(warm_path, config['input_size']).cuda()
    with torch.inference_mode():
        for _ in range(config['warmup_iterations']):
            network(warm, return_logits=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    started = time.perf_counter()
    try:
        for index, row in enumerate(tqdm(rows, desc='OOAL UMD inference', mininterval=5), 1):
            if sha(row['rgb']) != row['rgb_sha256']:
                raise ValueError(f'RGB changed: {row["sample_id"]}')
            torch.cuda.synchronize()
            sample_started = time.perf_counter()
            image = load_rgb(row['rgb'], config['input_size']).cuda()
            torch.cuda.synchronize()
            forward_started = time.perf_counter()
            with torch.inference_mode():
                logits = network(image, return_logits=True)
                height, width = row['size_wh'][1], row['size_wh'][0]
                logits = F.interpolate(logits, size=(height, width), mode='bilinear', align_corners=False)
                result = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
            torch.cuda.synchronize()
            completed = time.perf_counter()
            record = {'sample_id': row['sample_id'], 'model_and_resize_seconds': completed-forward_started,
                      'rgb_to_class_map_seconds': completed-sample_started}
            times.append(record)
            sample = out / 'samples' / row['sample_id']
            target = sample / 'result'
            target.mkdir(parents=True)
            Image.fromarray(result).save(target / 'class_map.png')
            for class_id, name in enumerate(config['class_names']):
                if class_id:
                    save_mask(target / 'masks' / f'{name}.png', result == class_id)
            write(sample / 'ooal_timings.json', record)
            write(sample / 'run.json', {'status': 'complete', 'reference_annotations_used': False})
            if index % 100 == 0 or index == len(rows):
                state['completed_samples'] = index
                write(out / 'run.json', state)
        state.update(status='complete', counts={'complete': len(rows)},
                     batch_wall_seconds=time.perf_counter()-started)
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', completed_samples=len(times))
        raise
    finally:
        write(out / 'run.json', state)
    write_jsonl(out / 'latency_per_image.jsonl', times)
    summary = {'gpu': torch.cuda.get_device_name(), 'batch_size': 1,
               'precision': config['precision'], 'warmup_iterations': config['warmup_iterations'],
               'input_size': config['input_size'], 'queries_per_forward': 'all 7 affordances + background',
               'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
               'scope': 'RGB file read/resize/normalize + H2D + model + native-size logit resize/argmax + D2H. Excludes hashing, saving, visualization and model loading.'}
    for key in ['model_and_resize_seconds', 'rgb_to_class_map_seconds']:
        values = [t[key] for t in times]
        summary[key] = {'count': len(values), 'mean': float(np.mean(values)),
                        'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95))}
    write(out / 'latency.json', summary)
    return state
