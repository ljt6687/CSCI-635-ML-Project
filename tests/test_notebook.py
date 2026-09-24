"""Regression checks extract the real helpers from the notebook, not copies."""
import ast
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nbformat
from sklearn.metrics import roc_curve, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
NB=nbformat.read(ROOT/'squirrel_detection_rfdetr_medium.ipynb',as_version=4)

def definitions(names,namespace):
    nodes=[]
    for cell in NB.cells:
        if cell.cell_type=='code':
            for node in ast.parse(cell.source).body:
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
                    nodes.append(node)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'notebook-helpers','exec'),namespace)
    return namespace

class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.ns=definitions({'box_iou','match_detections','detection_metrics','presence_roc'},
                            dict(np=np,math=math,roc_curve=roc_curve,roc_auc_score=roc_auc_score,CFG={'iou':.5}))
        self.gt=[[0,0,10,10]]
        self.pred=[dict(box=[0,0,10,10],score=.9)]
    def test_all_cells_compile(self):
        nbformat.validate(NB)
        for cell in NB.cells:
            if cell.cell_type=='code': compile(cell.source,'notebook','exec')
    def test_one_to_one_duplicates_and_misses(self):
        f=self.ns['match_detections']
        self.assertEqual(f(self.gt,self.pred,.5)[:3],(1,0,0))
        self.assertEqual(f(self.gt,self.pred*2,.5)[:3],(1,1,0))
        self.assertEqual(f(self.gt,[],.5)[:3],(0,0,1))
        self.assertEqual(f([],self.pred,.5)[:3],(0,1,0))
        self.assertEqual(f([],[],.5)[:3],(0,0,0))
    def test_confidence_order_and_iou(self):
        f=self.ns['match_detections']
        predictions=[dict(box=[20,20,30,30],score=.99),*self.pred]
        self.assertEqual(f(self.gt,predictions,.5)[:3],(1,1,0))
        self.assertEqual(f(self.gt,predictions,.95)[:3],(0,1,1))
        self.assertAlmostEqual(self.ns['box_iou']([0,0,10,10],[5,0,15,10]),1/3)
    def test_image_presence_roc(self):
        f=self.ns['presence_roc']
        self.assertIsNone(f([dict(gt=[],image_id=1)],{})[2])
        self.assertEqual(f([dict(gt=self.gt,image_id=1),dict(gt=[],image_id=2)],{1:self.pred})[2],1.)
    def test_empty_metrics_are_finite(self):
        result=self.ns['detection_metrics']([dict(gt=self.gt,image_id=1)],{},.5)
        self.assertEqual(result['f1'],0)
        self.assertEqual(result['fn'],1)
        self.assertIsNone(result['mean_matched_iou'])
    def test_redaction_fragmented_writes_and_signed_urls(self):
        import re
        with tempfile.TemporaryDirectory() as tmp:
            ns=definitions({'redact','assert_secret_free','SafeWriter'},dict(re=re,io=io,json=json,_SECRETS=['FAKE_SECRET_123'],RUN=Path(tmp)))
            target=io.StringIO(); writer=ns['SafeWriter'](target)
            writer.write('FAKE_SEC'); writer.write('RET_123 https://example.test/file?api_key=unknown\n'); writer.finish()
            self.assertNotIn('FAKE_SECRET_123',target.getvalue())
            self.assertNotIn('unknown',target.getvalue())
            self.assertIn('[REDACTED]',target.getvalue())
            with self.assertRaises(RuntimeError): ns['assert_secret_free']({'key':'FAKE_SECRET_123'})
    def test_coco_no_detections_and_perfect(self):
        import copy
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        doc={'images':[{'id':1,'width':20,'height':20}], 'categories':[{'id':7,'name':'SQUIRREL'}],
             'annotations':[{'id':1,'image_id':1,'category_id':7,'bbox':[0,0,10,10]}]}
        ns=definitions({'coco_metrics'},dict(COCO=COCO,COCOeval=COCOeval,splits={'test':doc}))
        with contextlib.redirect_stdout(io.StringIO()):
            empty=ns['coco_metrics']('test',[dict(image_id=1)],{1:[]})
            perfect=ns['coco_metrics']('test',[dict(image_id=1)],{1:self.pred})
        self.assertEqual(empty['AP50'],0.)
        self.assertAlmostEqual(perfect['AP50'],1.)
        self.assertIsNone(perfect['AP_large'])

if __name__=='__main__': unittest.main()

class DatasetAuditTests(unittest.TestCase):
    def test_leakage_exact_and_perceptual(self):
        import hashlib
        records=[
            dict(split='train',file_name='a.jpg',source='a',pixel_hash='same',phash='0000000000000000'),
            dict(split='valid',file_name='b.jpg',source='b',pixel_hash='same',phash='0000000000000000'),
            dict(split='test',file_name='c.jpg',source='c',pixel_hash='other',phash='0000000000000001'),
            dict(split='train',file_name='d.jpg',source='d',pixel_hash='different',phash='ffffffffffffffff')]
        ns=definitions({'find_leakage_pairs'},dict(np=np,hashlib=hashlib))
        pairs=ns['find_leakage_pairs'](records)
        self.assertEqual(sum(p['confirmed'] for p in pairs),1)
        self.assertEqual(sum(not p['confirmed'] for p in pairs),2)
        self.assertTrue(all(records[p['left']]['split']!=records[p['right']]['split'] for p in pairs))
    def test_corrupt_and_invalid_annotations_block(self):
        import hashlib,re
        import imagehash
        import pandas as pd
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); report=root/'reports'; report.mkdir()
            for split in ['train','valid','test']:
                folder=root/split; folder.mkdir()
                Image.new('RGB',(20,20),'gray').save(folder/'image.png')
                doc={'images':[dict(id=1,file_name='image.png',width=20,height=20)],
                     'categories':[dict(id=1,name='SQUIRREL')],
                     'annotations':[dict(id=1,image_id=1,category_id=1,bbox=[0,0,10,10])]}
                (folder/'_annotations.coco.json').write_text(json.dumps(doc))
            ns=definitions({'audit_dataset','xywh_to_xyxy'},dict(DATA=root,REPORT=report,json=json,Image=Image,
                hashlib=hashlib,imagehash=imagehash,re=re,math=math,pd=pd,
                save_json=lambda path,value:Path(path).write_text(json.dumps(value))))
            _,records,_=ns['audit_dataset'](); self.assertEqual(len(records),3)
            doc['annotations'][0]['bbox']=[19,0,10,10]
            (root/'train/_annotations.coco.json').write_text(json.dumps(doc))
            with self.assertRaisesRegex(RuntimeError,'integrity'): ns['audit_dataset']()
            doc['annotations'][0]['bbox']=[0,0,10,10]
            (root/'train/_annotations.coco.json').write_text(json.dumps(doc))
            (root/'valid/image.png').write_text('not an image')
            with self.assertRaisesRegex(RuntimeError,'integrity'): ns['audit_dataset']()
