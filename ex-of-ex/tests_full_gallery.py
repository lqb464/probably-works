import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from full_gallery import rank_scores, one_metrics, choices_for, select_choices, score_split, report, split_indices, special_tokens, run_addon
from scorers import build_specs


class FullGalleryTests(unittest.TestCase):
    def test_combined_cli_mode(self):
        from run_experiments import parse_args, apply_overrides
        with patch('sys.argv', ['run_experiments.py', '--config', 'example.yaml', '--with-full-gallery']):
            config = apply_overrides({}, parse_args())
        self.assertTrue(config['with_full_gallery'])
        self.assertFalse(config['full_gallery_only'])

    def test_special_eos_cls_are_independently_extracted(self):
        tokens = torch.tensor([[1,9,0],[1,2,9]])
        sequence = torch.arange(18,dtype=torch.float32).reshape(2,3,3)
        pixels = torch.ones(2,3)
        base = SimpleNamespace(encode_text=lambda x:(sequence,None),
                               encode_image=lambda x:(sequence,None),
                               text_projection=torch.eye(3),visual=SimpleNamespace(proj=torch.eye(3)))
        model = SimpleNamespace(base_model=base,eval=lambda:None)
        gf = SimpleNamespace(qfeats=sequence[torch.arange(2),tokens.argmax(-1)],gfeats=sequence[:,0],
                             qids=torch.arange(2),gids=torch.arange(2))
        q,g,audit = special_tokens(model,[(gf.qids,tokens)],[(gf.gids,pixels)],torch.device('cpu'),gf)
        self.assertTrue(audit['equivalent_to_global_at_1e-5'])
        self.assertEqual(q.shape,g.shape)

    def test_addon_end_to_end_holdout(self):
        n = 40
        gf = SimpleNamespace(qfeats=torch.eye(n),gfeats=torch.eye(n),qids=torch.arange(n),gids=torch.arange(n))
        hf = SimpleNamespace(text_features=torch.eye(n)[:,None,:],image_features=torch.eye(n)[:,None,:],
                             text_mask=torch.ones(n,1,dtype=torch.bool))
        config = {'validated':{'missing_validation':'test_identity_holdout','bootstrap_repetitions':2000},
                  'topk':[2],'fusion_weights':[.5]}
        with tempfile.TemporaryDirectory() as temp:
            with patch('full_gallery.special_tokens',return_value=(gf.qfeats,gf.gfeats,{'test':True})), \
                 patch('full_gallery.build_eval_split_loaders',return_value=(None,None)):
                result = run_addon(config,Path(temp),None,torch.device('cpu'),None,gf,hf,None,None,
                                   Path(temp)/'cache.pt',{'checkpoint_sha256':'fixture'})
            self.assertTrue(result.exists())
            self.assertTrue((result.parent/'summary.json').exists())
            self.assertTrue((result.parent/'selected_before_test.json').exists())

    def test_hidden_can_retrieve_outside_global_topk(self):
        global_score = np.array([.9,.8,.1])
        hidden = np.array([.1,.2,.99])
        self.assertEqual(rank_scores(global_score,hidden,1)[0],2)
        self.assertEqual(rank_scores(global_score,hidden,1,2)[0],1)
        np.testing.assert_array_equal(rank_scores(global_score,hidden,0),[0,1,2])
        np.testing.assert_array_equal(rank_scores(global_score,np.ones(3),1),[0,1,2])

    def test_exact_metrics(self):
        result = one_metrics(np.array([0,1,2]),1,np.array([0,1,1]))
        np.testing.assert_allclose(result,[0,1,1,(.5+2/3)/2,2/3])

    def test_selection_prefers_noop_on_tie(self):
        choices = choices_for(["maxsim"],[0,.5,1],[2])
        metrics = {c["key"]:np.ones((4,5)) for c in choices}
        selected = select_choices(choices,metrics)
        self.assertTrue(all(c["alpha"]==0 for c in selected if c["family"] in {"fusion_full","rerank_topk"}))

    def test_identity_holdout_disjoint(self):
        gf = SimpleNamespace(qids=torch.arange(100).repeat_interleave(2))
        si,ti,rows,_ = split_indices(None,gf,{"missing_validation":"test_identity_holdout"},42)
        self.assertEqual(len(ti),160)
        self.assertFalse(set(gf.qids[si].tolist()) & set(gf.qids[ti].tolist()))
        self.assertEqual(len(rows),100)

    def test_streamed_scoring_and_bootstrap(self):
        gf = SimpleNamespace(qfeats=torch.eye(4),gfeats=torch.eye(4),qids=torch.arange(4),gids=torch.arange(4))
        hf = SimpleNamespace(text_features=torch.eye(4)[:,None,:],image_features=torch.eye(4)[:,None,:],
                             text_mask=torch.ones(4,1,dtype=torch.bool))
        specs = build_specs([{"name":"maxsim","kind":"maxsim"}])
        choices = choices_for(["maxsim"],[0,.5,1],[2])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            metrics,top = score_split(gf,hf,(gf.qfeats,gf.gfeats),np.arange(4),specs,choices,
                                      torch.device("cpu"),2,output,"test",True)
            for c in choices:
                np.testing.assert_allclose(metrics[c["key"]],1)
                np.testing.assert_array_equal(top[c["key"]],np.arange(4))
            np.testing.assert_allclose(np.load(output/"test__maxsim__similarity.npy"),np.eye(4))
            rows = report(select_choices(choices,metrics),metrics,np.arange(4),{"bootstrap_repetitions":2000})
            self.assertTrue(all(r["r1"]==100 and r["delta_r1"]==0 for r in rows))
            self.assertTrue(all(r["r1_bootstrap_valid"]==2000 for r in rows))


if __name__ == "__main__":
    unittest.main()
