# ST-LLM-Attn

## 安装
安装相关依赖
```bash
conda create -n ST-LLM-Attn python=3.10
conda activate ST-LLM-Attn
git clone https://github.com/Hzzz123rfefd/ST-LLM-Attn.git
pip install -r requirement.txt
```

## 使用方式
### 数据集
首先你需要准备时序数据集，由于保密原因，此处数据集用dataset1表示，你需要放入dataset文件夹下，因此你的目录结构为
- eticn/
  - datasets/
    - your_datasets/
      - file1
      - file2
      - process.py

你需要自己根据你的数据格式完成process.py，该函数的作用应该包括：
* 保存进站流为npy文件，命名inflow.npy，shape = （n，t），其中n为站点，t为时间序列
* 保存出站流为npy文件，命名outflow.npy，shape = （n，t），其中n为站点，t为时间序列
* 保存预测npy文件，放入文件夹dataset/your_datasets/npy_data
该文件夹中保存inflow_{index}.npy，和outflow_{index}.npy维度为（n，t‘ + 1），其中n为站点，t‘为需要多少个时间步去预测下一个时间步
* 保存预测格式jsonl，放入文件夹your_datasets_trainning
该文件夹中保存train.jsonl、test.jsonl、val.jsonl， 每个jsonl格式如下：
```jsonl
{"inflow_path": "dataset/your_datasets/npy_data\\inflow_0.npy", "outflow_path": "dataset/your_datasets/npy_data\\outflow_0.npy"}
{"inflow_path": "dataset/your_datasets/npy_data\\inflow_1.npy", "outflow_path": "dataset/your_datasets/npy_data\\outflow_1.npy"}
.....
```

### 模型训练
你可以在`config\st_llm_attn.yml`中修改训练、模型相关配置参数，然后运行以下脚本开始模型训练
```bash
python train.py --model_config_path config/st_llm_attn.yml
```

### 模型评估
你可以运行以下脚本开始评估模型，评价指标包括RMSE、MAE 、WMAPE
```bash
python example\eval.py --model_config_path config/st_llm_attn.yml --data_path gaotie_trainning/test.jsonl --model_path saved_model/st_llm_attn
```

### 模型推理
你可以运行以下脚本开始进行模型推理，这依赖于你之前生成的inflow.npy和outflow.npy
```bash
python example\inference.py --model_config_path config/st_llm_attn.yml --inflow_path dataset/gaotie/inflow.npy --outflow_path dataset/gaotie/outflow.npy --model_path saved_model/st_llm_attn --step 7 --save_dir result/
```

### 结果可视化
首先你需要进行模型推理，然后运行以下脚本即可进行结果可视化
python example\plot.py --save_dir result/
```
