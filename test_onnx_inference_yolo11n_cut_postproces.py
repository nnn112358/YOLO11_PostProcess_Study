import onnxruntime
import numpy as np

def run_onnx_inference(model_path, input_arrays):
    """
    ONNXモデルを実行して出力を取得する関数
    
    Args:
        model_path (str): ONNXモデルファイルのパス
        input_arrays (list): 3つの入力配列のリスト
        
    Returns:
        numpy.ndarray: モデルの出力
    """
    # セッションの初期化
    session = onnxruntime.InferenceSession(model_path)
    
    # 入力名と出力名の設定
    input_names = [
        "/model.23/Concat_output_0",
        "/model.23/Concat_1_output_0", 
        "/model.23/Concat_2_output_0"
    ]
    output_names = ["output0"]
    
    # 入力データの辞書を作成
    input_dict = {}
    for name, array in zip(input_names, input_arrays):
        input_dict[name] = array
    
    # 推論の実行
    outputs = session.run(
        output_names,
        input_dict
    )
    
    # 単一の出力を返す
    return outputs[0]

# 使用例
if __name__ == "__main__":
    # モデルパスの設定
    model_path = "yolo11n_cut_post.onnx"
    
    # サンプル入力データの作成
    # 注意: 実際の入力サイズはモデルの要件に合わせて調整してください
    input1 = np.random.random((1, 144, 80, 80)).astype(np.float32)
    input2 = np.random.random((1, 144, 40, 40)).astype(np.float32)
    input3 = np.random.random((1, 144, 20, 20)).astype(np.float32)
    
    # 入力配列のリストを作成
    input_arrays = [input1, input2, input3]
    
    # モデルの実行
    output = run_onnx_inference(model_path, input_arrays)
    
    # 出力の形状を表示
    print("Output shape:", output.shape)