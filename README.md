## 目的

YOLO11の量子化のための、Post処理の検証用。


### 

yolo11n.onnx をyolo11n_cut_backbone.onnx + yolo11n_cut_postproces.onnxに分割する 

![image](https://github.com/user-attachments/assets/026d4b91-19f4-4d60-b2b2-4f8b3f7097b4)

```
python yolo11_cut-onnx.py
# yolo11n.onnx -> yolo11n_cut_backbone.onnx + yolo11n_cut_postproces.onnx
```

```
python test_onnx_inference_yolo11n_cut__backbone.py
python test_onnx_inference_yolo11n_cut_all.py
python test_onnx_inference_yolo11n_cut_postproces.py
```


```
$ python yolo11_inference_onnx.py

処理時間の内訳:
前処理時間: 0.007 秒
outputs推論時間: 0.126 秒
後処理時間: 0.002 秒
合計時間: 0.135 秒
推論時間: 0.136 秒
検出: person, 信頼度: 0.77, 位置: (231, 449, 292, 584)
検出: person, 信頼度: 0.71, 位置: (420, 452, 470, 551)
検出: person, 信頼度: 0.70, 位置: (695, 438, 746, 575)
検出: person, 信頼度: 0.66, 位置: (628, 453, 676, 573)
検出: person, 信頼度: 0.65, 位置: (484, 444, 534, 562)
検出: person, 信頼度: 0.59, 位置: (335, 452, 377, 569)
検出: person, 信頼度: 0.51, 位置: (373, 442, 429, 580)
検出: person, 信頼度: 0.48, 位置: (546, 440, 590, 569)
検出: person, 信頼度: 0.47, 位置: (0, 469, 48, 598)
検出: person, 信頼度: 0.44, 位置: (177, 445, 226, 554)
検出: person, 信頼度: 0.43, 位置: (761, 443, 798, 576)
$ python yolo11_inference_split_onnx.py

処理時間の内訳:
前処理時間: 0.018 秒
バックボーン推論時間: 0.168 秒
ポストプロセスモデル時間: 0.018 秒
後処理時間: 0.001 秒
合計時間: 0.205 秒
推論時間: 0.206 秒
検出: person, 信頼度: 0.77, 位置: (231, 449, 292, 584)
検出: person, 信頼度: 0.71, 位置: (420, 452, 470, 551)
検出: person, 信頼度: 0.70, 位置: (695, 438, 746, 575)
検出: person, 信頼度: 0.66, 位置: (628, 453, 676, 573)
検出: person, 信頼度: 0.65, 位置: (484, 444, 534, 562)
検出: person, 信頼度: 0.59, 位置: (335, 452, 377, 569)
検出: person, 信頼度: 0.51, 位置: (373, 442, 429, 580)
検出: person, 信頼度: 0.48, 位置: (546, 440, 590, 569)
検出: person, 信頼度: 0.47, 位置: (0, 469, 48, 598)
検出: person, 信頼度: 0.44, 位置: (177, 445, 226, 554)
検出: person, 信頼度: 0.43, 位置: (761, 443, 798, 576)
```

![image](https://github.com/user-attachments/assets/3f8e983d-6a30-4d82-bc5f-ef343b8d6ab7)

写真は、ぱくたそオリジナルのフリー素材です。
https://www.pakutaso.com/20190231052post-19700.html

