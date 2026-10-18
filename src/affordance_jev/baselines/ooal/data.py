"""Author-selected UMD one-shot images, with instance/class leakage checks."""
import io
from collections import Counter
from pathlib import Path
import random
import tarfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from ...common import read_jsonl, sha, write, write_jsonl
from ...datasets.umd import load_annotation, read_split


def prepare(config):
    root = Path(config['dataset_root'])
    archive = Path(config['one_shot_archive'])
    if sha(archive) != config['one_shot_archive_sha256']:
        raise ValueError('Author one-shot archive checksum mismatch')
    split = read_split(root / 'novel_split.txt')
    test = read_jsonl(config['test_manifest'])
    test_instances = {r['object_instance'] for r in test}
    if test_instances != {name for name, role in split.items() if role == 'test'}:
        raise ValueError('Test manifest does not cover the official novel/test instances')
    rows = []
    archive_rgb_identical = 0
    with tarfile.open(archive) as stream:
        labels = sorted(m for m in stream.getnames()
                        if m.startswith('train_novel_one_pick/') and m.endswith('_label.png'))
        for name in labels:
            stem = Path(name).name.removesuffix('_label.png')
            instance, frame = stem.rsplit('_', 1)
            if split[instance] != 'train' or instance in test_instances:
                raise ValueError(f'Train/test leakage: {instance}')
            source = root / 'tools' / instance
            rgb_path, annotation = source / (stem + '_rgb.jpg'), source / (stem + '_label.mat')
            gt, kind = load_annotation(annotation)
            author_gt = np.asarray(Image.open(io.BytesIO(stream.extractfile(name).read())))
            if not np.array_equal(gt, author_gt):
                raise ValueError(f'Native labels differ from author novel training labels: {name}')
            author_rgb_path = name.replace('_label.png', '_rgb.jpg')
            author_rgb = np.asarray(Image.open(io.BytesIO(stream.extractfile(author_rgb_path).read())).convert('RGB'))
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert('RGB'))
                size = list(image.size)
            equal_rgb = np.array_equal(rgb, author_rgb)
            if not equal_rgb:
                raise ValueError(f'RGB pixels differ from author training image: {name}')
            archive_rgb_identical += int(equal_rgb)
            rows.append({
                'sample_id': f'{instance}/{frame}', 'object_instance': instance,
                'frame_index': int(frame), 'rgb': str(rgb_path.resolve()),
                'annotation': str(annotation.resolve()), 'gt_type': kind,
                'size_wh': size, 'rgb_sha256': sha(rgb_path), 'annotation_sha256': sha(annotation),
                'author_label_member': name, 'author_rgb_pixels_match': equal_rgb,
                'split': 'novel/train',
            })
    expected = {name for name, role in split.items() if role == 'train'}
    instances = {r['object_instance'] for r in rows}
    if len(rows) != 76 or instances != expected or len(instances) != len(rows):
        raise ValueError('Expected exactly one author-selected image per each of 76 train instances')
    category = lambda names: {n.rsplit('_', 1)[0] for n in names}
    if category(instances) & category(test_instances):
        raise ValueError('Training and test object categories overlap')
    if {r['rgb_sha256'] for r in rows} & {r['rgb_sha256'] for r in test}:
        raise ValueError('Training and test RGB files overlap')
    output = Path(config['train_manifest'])
    if output.exists():
        if read_jsonl(output) != rows:
            raise FileExistsError(f'Existing manifest differs: {output}')
    else:
        write_jsonl(output, rows)
    report = {
        'training_samples': len(rows), 'training_instances': len(instances),
        'training_categories': sorted(category(instances)),
        'annotation_types': dict(Counter(r['gt_type'] for r in rows)),
        'author_rgb_pixels_match': archive_rgb_identical,
        'author_labels_match_original_dominant': len(rows),
        'test_samples': len(test), 'test_instances': len(test_instances),
        'test_categories': sorted(category(test_instances)),
        'instance_overlap': [], 'category_overlap': [], 'rgb_file_hash_overlap': [],
        'archive_sha256': sha(archive), 'split_sha256': sha(root / 'novel_split.txt'),
        'train_manifest_sha256': sha(output), 'test_manifest_sha256': sha(config['test_manifest']),
        'label_policy': 'Original 0..7 dominant labels. Novel training contains no ladle.',
    }
    write(output.with_suffix('.meta.json'), report)
    return report


def normalize(image):
    return TF.normalize(TF.to_tensor(image), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class OneShot(Dataset):
    def __init__(self, manifest, resize_size, input_size, flip_probability=0.5):
        self.rows = read_jsonl(manifest)
        self.input_size, self.flip_probability = input_size, flip_probability
        self.images, self.labels = [], []
        for row in self.rows:
            if sha(row['rgb']) != row['rgb_sha256'] or sha(row['annotation']) != row['annotation_sha256']:
                raise ValueError(f'Training input changed: {row["sample_id"]}')
            image = Image.open(row['rgb']).convert('RGB')
            gt, _ = load_annotation(row['annotation'])
            self.images.append(image.resize((resize_size, resize_size), Image.Resampling.BILINEAR))
            self.labels.append(Image.fromarray(gt.astype(np.int32), mode='I').resize(
                (resize_size, resize_size), Image.Resampling.NEAREST))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        width, height = image.size
        side = self.input_size
        top = int(torch.randint(0, height - side + 1, (1,)).item())
        left = int(torch.randint(0, width - side + 1, (1,)).item())
        box = (left, top, left + side, top + side)
        image, label = image.crop(box), label.crop(box)
        if random.random() < self.flip_probability:
            image, label = TF.hflip(image), TF.hflip(label)
        return normalize(image), torch.from_numpy(np.asarray(label).copy()).long()


def load_rgb(path, size):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
    return normalize(rgb.resize((size, size), Image.Resampling.BILINEAR)).unsqueeze(0)
