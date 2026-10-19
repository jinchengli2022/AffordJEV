"""Official RGB + known object category + action prompt; no mask annotation input."""
import json
import logging
import os
import re
import time

import numpy as np
from PIL import Image


OFFICIAL_TEMPLATE = (
    'Please answer "{Question}" with bboxs and points.'
    'Analyze the functional properties of specific parts of each object in the image and carefully find all the part(s) that matches the problem.'
    'Output the thinking process in <think> </think>, rethinking process in <rethink> </rethink> and final answer in <answer> </answer> tags.'
    'Output the bbox(es) and point(s) and affordance tpye(s) inside the interested object(s) in JSON format.'
    'i.e., <think> thinking process here </think>'
    '<rethink> rethinking process here </rethink>'
    '<answer>{Answer}</answer>'
)
ANSWER_EXAMPLE = ('[{"bbox_2d": [10,100,200,210], "point_2d": [30,110], "affordance": "hold"}, '
                  '{"bbox_2d": [225,296,706,786], "point_2d": [302,410], "affordance": "grasp"}]')


def prompt_for(category, action):
    query = f'What part of the {category} should we interact with in order to {action} it?'
    return OFFICIAL_TEMPLATE.format(Question=query.lower().strip('.'), Answer=ANSWER_EXAMPLE)


def parse_answer(text, side, width, height):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S)
    if not match:
        raise ValueError("missing answer tags")
    items = json.loads(match.group(1))
    if not isinstance(items, list):
        raise ValueError("answer must be a JSON list")
    regions = []
    for item in items:
        box = np.asarray(item["bbox_2d"], dtype=np.float64)
        point = np.asarray(item["point_2d"], dtype=np.float64)
        if box.shape != (4,) or point.shape != (2,) or not np.isfinite(box).all() or not np.isfinite(point).all():
            raise ValueError("invalid coordinates")
        # Official code scales and rounds the 840-coordinate output.
        box = np.floor(np.clip(box, 0, side - 1) * [width, height, width, height] / side + 0.5)
        point = np.floor(np.clip(point, 0, side - 1) * [width, height] / side + 0.5)
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("degenerate box")
        regions.append({"bbox_original": box.tolist(), "point_original": point.tolist(),
                        "generated_affordance": item.get("affordance")})
    return regions


def merge_logits(logits):
    """Inputs ordered native IDs 1..7; threshold and tie rule are GT-free."""
    best = np.argmax(logits, axis=0)
    result = (best + 1).astype(np.uint8)
    result[np.max(logits, axis=0) <= 0] = 0
    return result


class Predictor:
    def __init__(self, model_path, sam_checkpoint, config, definitions):
        import torch
        from transformers import AutoProcessor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['VLLM_USE_V1'] = '1'
        os.environ['VLLM_HOST_IP'] = '127.0.0.1'
        os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo')
        from vllm import LLM, SamplingParams
        self.torch, self.config = torch, config
        self.classes = sorted((c for c in definitions['classes'] if c['id']), key=lambda c: c['id'])
        assert [c['id'] for c in self.classes] == list(range(1, 8))
        assert config['image_side'] == 840 and config['mask_threshold'] == 0
        torch.set_num_threads(config['num_threads'])
        torch.manual_seed(config['seed'])
        side = config['image_side']
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False,
                                                       min_pixels=side**2, max_pixels=side**2)
        self.prompts = {'official_template': OFFICIAL_TEMPLATE, 'answer_example': ANSWER_EXAMPLE,
                        'query': 'What part of the {category} should we interact with in order to {action} it?'}
        self.model = LLM(model=str(model_path), dtype=config['dtype'], seed=config['seed'],
            tensor_parallel_size=1, max_model_len=config['max_model_len'],
            gpu_memory_utilization=config['gpu_memory_utilization'],
            max_num_seqs=config['max_num_seqs'], max_num_batched_tokens=512,
            limit_mm_per_prompt={'image': 1, 'video': 0},
            mm_processor_kwargs={'min_pixels': side**2, 'max_pixels': side**2},
            enable_prefix_caching=True, enforce_eager=False,
            compilation_config={'cudagraph_capture_sizes': [1, 2, 4, 7, 14, 21]}, disable_log_stats=True)
        self.sampling = SamplingParams(temperature=0, max_tokens=config['max_new_tokens'],
                                       seed=config['seed'], skip_special_tokens=True)
        logging.getLogger().setLevel(logging.WARNING)
        self.sam = SAM2ImagePredictor(build_sam2(config['sam_config'], str(sam_checkpoint), device='cuda'))

    def predict(self, image, category):
        return self.predict_many([image], [category])[0]

    def predict_many(self, images, categories, actions_per_image=None):
        torch = self.torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        images = [image.convert('RGB') for image in images]
        assert len(images) == len(categories)
        conditional = actions_per_image is not None
        if conditional:
            if len(actions_per_image) != len(images):
                raise ValueError('One explicit action list per RGB is required')
            lookup = {c['name']: c for c in self.classes}
            groups = []
            for actions in actions_per_image:
                if not actions or len(set(actions)) != len(actions) or not set(actions) <= lookup.keys():
                    raise ValueError('Invalid or duplicate requested actions')
                groups.append([lookup[action] for action in actions])
        else:
            groups = [self.classes for _ in images]
        side = self.config['image_side']
        requests = []
        for image, category, classes in zip(images, categories, groups):
            resized = image.resize((side, side), Image.Resampling.BILINEAR)
            chat = [self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'image'},
                 {'type': 'text', 'text': prompt_for(category, c['name'])}]}],
                tokenize=False, add_generation_prompt=True) for c in classes]
            requests.extend({'prompt': p, 'multi_modal_data': {'image': resized}} for p in chat)
        pre = time.perf_counter()
        outputs = self.model.generate(requests, self.sampling, use_tqdm=False)
        torch.cuda.synchronize()
        generated = time.perf_counter()
        results = []
        offset = 0
        for image, category, classes in zip(images, categories, groups):
            results.append(self.segment(image, outputs[offset:offset+len(classes)], category,
                                        classes=classes, merge=not conditional))
            offset += len(classes)
        assert offset == len(outputs)
        end = time.perf_counter()
        for result in results:
            timing = result[3]
            timing.update(preprocess_seconds_amortized=(pre-start)/len(images),
                          generation_seconds_amortized_per_rgb=(generated-pre)/len(images),
                          batch_wall_seconds=end-start,
                          rgb_to_requested_masks_seconds=(end-start)/len(images),
                          seconds_kind='serial latency' if len(images)==1 else 'amortized batch throughput cost',
                          image_batch=len(images), operation_batch=len(requests),
                          conditional_actions_only=conditional,
                          model_load_and_disk_io_excluded=True)
            if not conditional:
                timing.update(generation_all_seven_seconds_amortized=(generated-pre)/len(images),
                              rgb_to_all_classes_seconds=(end-start)/len(images))
        return results

    def segment(self, image, outputs, category, classes=None, merge=True):
        torch = self.torch
        width, height = image.size
        side = self.config['image_side']
        start = time.perf_counter()
        logits = np.full((7, height, width), -np.inf, dtype=np.float32)
        records = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            self.sam.set_image(image)
            torch.cuda.synchronize()
            encoded = time.perf_counter()
            for cls, output in zip(self.classes if classes is None else classes, outputs):
                i = cls['id'] - 1
                response = output.outputs[0]
                record = {'class_id': cls['id'], 'action': cls['name'], 'response': response.text,
                          'object_category': category, 'prompt': prompt_for(category, cls['name']),
                          'generated_tokens': len(response.token_ids), 'finish_reason': response.finish_reason,
                          'regions': [], 'error': None}
                try:
                    regions = parse_answer(response.text, side, width, height)
                    for region in regions:
                        masks, scores, _ = self.sam.predict(
                            point_coords=np.array([region['point_original']]), point_labels=np.array([1]),
                            box=np.array(region['bbox_original']), multimask_output=True, return_logits=True)
                        chosen = int(np.argmax(scores))
                        score_map = masks[chosen]
                        if not np.isfinite(score_map).all():
                            raise ValueError('nonfinite SAM2 logits')
                        logits[i] = np.maximum(logits[i], score_map)
                        record['regions'].append({**region, 'sam_scores': scores.tolist(),
                                                  'selected_mask': chosen,
                                                  'mask_pixels': int((score_map > 0).sum())})
                except Exception as exc:
                    record['error'] = f'{type(exc).__name__}: {exc}'
                    record['regions'] = []
                    logits[i] = -np.inf
                records.append(record)
        torch.cuda.synchronize()
        segmented = time.perf_counter()
        class_map = merge_logits(logits) if merge else None
        end = time.perf_counter()
        timing = {'sam_encoder_seconds': encoded-start,
                  'sam_decoder_seconds': segmented-encoded, 'merge_seconds': end-segmented,
                  'model_load_and_disk_io_excluded': True}
        if merge:
            timing['sam_decoder_all_seven_seconds'] = segmented-encoded
        return class_map, logits, records, timing
