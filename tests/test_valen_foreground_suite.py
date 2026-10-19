"""Regression checks for per-operation candidates and unchanged functional decisions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from affordance_jev.common import read, write, save_mask, load_mask, sha
from affordance_jev.experiments.valen_foreground.workers import stage_functions, foreground_path, function_path
from affordance_jev.experiments.valen_foreground.suite_report import METHODS


class FunctionalSuite(unittest.TestCase):
    def test_five_methods_use_each_operations_new_parts_and_keep_original_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = root/'rgb.png'
            Image.new('RGB', (12, 10), 'gray').save(rgb)
            paths = {}
            for action, x in [('grasp', 1), ('cut', 7)]:
                q = {'sample_id': 'tool/1', 'query_id': f'tool/1/{action}', 'action': action,
                     'class_id': 1 if action == 'grasp' else 2, 'rgb': str(rgb),
                     'rgb_sha256': sha(rgb), 'size_wh': [12, 10], 'object_category': 'tool'}
                mask = np.zeros((10, 12), bool)
                mask[2:7, x:x+3] = True
                source = root/'parts'/action
                save_mask(source/'candidate.png', mask)
                fg = foreground_path(root, q)
                save_mask(fg/'selected_mask.png', mask)
                write(fg/'selection.json', {'reason': 'selected', 'mask_sha256': sha(fg/'selected_mask.png')})
                write(fg/'run.json', {'status': 'complete'})
                write(fg/'parts.json', {'status': 'ready', 'path': str(source.relative_to(root))})
                candidates = [{'id': 1, 'mask': 'candidate.png'}]
                def prepare_query(assets, out, query, cs, definitions, method):
                    out.mkdir(parents=True)
                    folder = out/'actual_request'
                    write(folder/'valen_response.json', {'usage': {'output_tokens': 0}, 'answers': {
                        'region_01': {'type': 'choice', 'probabilities': {'yes': .8, 'no': .2}}}})
                    write(folder/'valen_runtime.json', {'timings': {'predict_seconds': .1,
                        'compiled_input_save_seconds_inside_predict': 0}})
                    return [{'folder': str(folder), 'candidates': cs}]
                for method in METHODS:
                    with patch('affordance_jev.experiments.valen_foreground.workers.prepare_assets', return_value=candidates) as assets, \
                         patch('affordance_jev.experiments.valen_foreground.workers.prepare_query', side_effect=prepare_query) as requests:
                        status = stage_functions(Mock(), root, q, {}, method)
                    self.assertEqual(status, 'predicted')
                    self.assertEqual(assets.call_args.args[0], source)
                    self.assertEqual(assets.call_args.args[2], method)
                    self.assertEqual(requests.call_args.args[-1], method)
                    out = function_path(root, q, method)
                    self.assertTrue(np.array_equal(load_mask(out/'mask.png'), mask))
                    self.assertTrue(read(out/'decisions.json')[0]['selected'])
                    paths[method, action] = out
            self.assertEqual(len(set(paths.values())), 10)

    def test_empty_foreground_is_valid_empty_prediction_without_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = root/'rgb.png'
            Image.new('RGB', (12, 10)).save(rgb)
            q = {'sample_id': 'tool/1', 'query_id': 'tool/1/grasp', 'action': 'grasp',
                 'rgb': str(rgb), 'size_wh': [12, 10]}
            fg = foreground_path(root, q)
            save_mask(fg/'selected_mask.png', np.zeros((10, 12), bool))
            write(fg/'selection.json', {'reason': 'no_target', 'mask_sha256': sha(fg/'selected_mask.png')})
            write(fg/'run.json', {'status': 'complete'})
            write(fg/'parts.json', {'status': 'skipped', 'path': None})
            model = Mock()
            for method in METHODS:
                self.assertEqual(stage_functions(model, root, q, {}, method), 'no_target')
                self.assertFalse(load_mask(function_path(root, q, method)/'mask.png').any())
                self.assertEqual(read(function_path(root, q, method)/'run.json')['status'], 'complete')
            model.predict.assert_not_called()


if __name__ == '__main__':
    unittest.main()
