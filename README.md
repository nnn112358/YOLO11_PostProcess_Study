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
python　yolo11_inference.py
```

```
python　yolo11_inference_split_onnx.py
```
