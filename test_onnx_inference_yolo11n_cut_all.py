import onnxruntime
import numpy as np
from typing import Union, List, Dict, Tuple

def create_inference_session(model_path: str) -> Tuple[onnxruntime.InferenceSession, Dict]:
    """
    ONNXモデルのInferenceSessionを作成し、モデル情報を取得
    
    Args:
        model_path (str): モデルファイルのパス
        
    Returns:
        Tuple[onnxruntime.InferenceSession, Dict]: (セッション, モデル情報の辞書)
    """
    session = onnxruntime.InferenceSession(model_path)
    
    # モデルの入出力情報を取得
    model_info = {
        'input_names': [input.name for input in session.get_inputs()],
        'output_names': [output.name for output in session.get_outputs()],
        'input_shapes': [input.shape for input in session.get_inputs()],
        'output_shapes': [output.shape for output in session.get_outputs()]
    }
    
    return session, model_info

def run_inference(
    session: onnxruntime.InferenceSession,
    inputs: Union[np.ndarray, List[np.ndarray]],
    input_names: List[str],
    output_names: List[str]
) -> List[np.ndarray]:
    """
    ONNXモデルの推論を実行
    
    Args:
        session (onnxruntime.InferenceSession): 推論セッション
        inputs (Union[np.ndarray, List[np.ndarray]]): 入力データ
        input_names (List[str]): 入力テンソル名のリスト
        output_names (List[str]): 出力テンソル名のリスト
        
    Returns:
        List[np.ndarray]: 出力テンソルのリスト
    """
    # 入力データの準備
    if isinstance(inputs, np.ndarray):
        input_dict = {input_names[0]: inputs}
    else:
        input_dict = dict(zip(input_names, inputs))
    
    # 推論実行
    outputs = session.run(output_names, input_dict)
    return outputs

def run_onnx_inference(model_path: str, inputs: Union[np.ndarray, List[np.ndarray]], stage_name: str = "") -> Tuple[List[np.ndarray], Dict]:
    """
    ONNXモデルの推論処理を実行する統合関数
    
    Args:
        model_path (str): モデルファイルのパス
        inputs (Union[np.ndarray, List[np.ndarray]]): 入力データ
        stage_name (str): 実行ステージの名前（ログ出力用）
        
    Returns:
        Tuple[List[np.ndarray], Dict]: (出力テンソルのリスト, モデル情報の辞書)
    """
    # セッション作成とモデル情報取得
    session, model_info = create_inference_session(model_path)
    
    # ログ出力
    if stage_name:
        print(f"\n=== {stage_name} ===")
        print(f"Input names: {model_info['input_names']}")
        print(f"Output names: {model_info['output_names']}")
    
    # 推論実行
    outputs = run_inference(
        session,
        inputs,
        model_info['input_names'],
        model_info['output_names']
    )
    
    return outputs, model_info

def display_tensor_values(name: str, value: np.ndarray, max_elements: int = 5):
    """
    テンソルの値を表示する補助関数
    
    Args:
        name (str): テンソル名
        value (np.ndarray): テンソル値
        max_elements (int): 各次元で表示する最大要素数
    """
    print(f"\n{name}:")
    print(f"Shape: {value.shape}")
    print(f"Min value: {np.min(value)}")
    print(f"Max value: {np.max(value)}")
    print(f"Mean value: {np.mean(value)}")
    
    # 行列の値を表示
    print("Values (first few elements):")
    if len(value.shape) == 1:  # 1次元配列
        print(value[:max_elements])
    elif len(value.shape) == 2:  # 2次元配列
        print(value[:max_elements, :max_elements])
    elif len(value.shape) == 3:  # 3次元配列
        print(value[:max_elements, :max_elements, :max_elements])
    elif len(value.shape) == 4:  # 4次元配列（バッチ, チャネル, 高さ, 幅）
        print(value[0, :max_elements, :max_elements, :max_elements])  # 最初のバッチのみ表示
    else:
        print("Tensor has more than 4 dimensions, showing first slice")
        print(value[(0,) * (len(value.shape)-2) + (slice(max_elements), slice(max_elements))])


def run_yolo_pipeline_simple(
    cut_model_path: str,
    input_image: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    Complete pipeline: 両方のモデルを順番に実行
    
    Args:
        cut_model_path (str): First stageモデルのパス
        post_model_path (str): Second stageモデルのパス
        input_image (numpy.ndarray): 入力画像データ (1, 3, 640, 640)
        verbose (bool): 詳細なログを出力するかどうか
        
    Returns:
        numpy.ndarray: 最終出力
    """
    # First stage実行
    final_outputs, cut_info = run_onnx_inference(
        cut_model_path,
        input_image,
        "First Stage (Cut Model)" if verbose else ""
    )
    
    if verbose:
        print("\n=== Model Information ===")
        print("\nCut Model:")
        print(f"Input shapes: {cut_info['input_shapes']}")
        print(f"Output shapes: {cut_info['output_shapes']}")
        

        
        print("\n=== Output Shapes ===")
        print("Intermediate outputs:")
        for i, output in enumerate(final_outputs):
            print(f"Stage 1 Output {i}: {output.shape}")
        print(f"Final output: {final_outputs[0].shape}")
        np.set_printoptions(suppress=True, precision=3)
        print(final_outputs[0])

 
    return final_outputs[0]


def run_yolo_pipeline(
    cut_model_path: str,
    post_model_path: str,
    input_image: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    Complete pipeline: 両方のモデルを順番に実行
    
    Args:
        cut_model_path (str): First stageモデルのパス
        post_model_path (str): Second stageモデルのパス
        input_image (numpy.ndarray): 入力画像データ (1, 3, 640, 640)
        verbose (bool): 詳細なログを出力するかどうか
        
    Returns:
        numpy.ndarray: 最終出力
    """
    # First stage実行
    intermediate_outputs, cut_info = run_onnx_inference(
        cut_model_path,
        input_image,
        "First Stage (Cut Model)" if verbose else ""
    )
    
    # Second stage実行
    final_outputs, post_info = run_onnx_inference(
        post_model_path,
        intermediate_outputs,
        "Second Stage (Post Model)" if verbose else ""
    )
    
    if verbose:
        print("\n=== Model Information ===")
        print("\nCut Model:")
        print(f"Input shapes: {cut_info['input_shapes']}")
        print(f"Output shapes: {cut_info['output_shapes']}")
        
        print("\nPost Model:")
        print(f"Input shapes: {post_info['input_shapes']}")
        print(f"Output shapes: {post_info['output_shapes']}")
        
        print("\n=== Output Shapes ===")
        print("Intermediate outputs:")
        for i, output in enumerate(intermediate_outputs):
            print(f"Stage 1 Output {i}: {output.shape}")
        print(f"Final output: {final_outputs[0].shape}")
        np.set_printoptions(suppress=True, precision=3)
        print(final_outputs[0])

 
    return final_outputs[0]

if __name__ == "__main__":
    # モデルパスの設定
    cut_model_path = "yolo11n_cut_backbone.onnx"
    post_model_path = "yolo11n_cut_postproces.onnx"
    
    # サンプル入力画像の作成 (1, 3, 640, 640)
    sample_input = np.random.random((1, 3, 640, 640)).astype(np.float32)
    
    try:
        # パイプライン実行
        final_output = run_yolo_pipeline(
            cut_model_path,
            post_model_path,
            sample_input,
            verbose=True
        )
        print("\nPipeline execution completed successfully!")
        
    except Exception as e:
        print(f"\nError during pipeline execution: {str(e)}")

    full_model_path = "yolo11n.onnx"

    try:
        # パイプライン実行
        final_output = run_yolo_pipeline_simple(
            full_model_path,
            sample_input,
            verbose=True
        )
        print("\nPipeline execution completed successfully!")
        
    except Exception as e:
        print(f"\nError during pipeline execution: {str(e)}")






