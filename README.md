## 目的

YOLO11の量子化のための、Post処理の検証用。


yolo11n.onnx -> yolo11n_cut_backbone.onnx + yolo11n_cut_postproces.onnx

```
python yolo11_cut-onnx.py
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
