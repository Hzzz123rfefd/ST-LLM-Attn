# ST-LLM-Attn
## 📦 安装
```bash
# 创建虚拟环境（需提前安装 Anaconda/Miniconda）
conda create -n ST-LLM-Attn python=3.10 -y
conda activate ST-LLM-Attn
# 克隆仓库并安装依赖
git clone https://github.com/Hzzz123rfefd/ST-LLM-Attn.git
cd ST-LLM-Attn
pip install -r requirements.txt  # 注意文件名应为复数形式
```
## 🚀 使用指南
### 数据集准备
#### 目录结构
```text
ST-LLM-Attn/
├── datasets/
│   └── your_dataset/       # 自定义数据集名称
│       ├── file1           # 原始数据文件
│       ├── file2
│       └── process.py      # 数据预处理脚本
```
#### 预处理脚本要求
在 `process.py` 中需完成以下操作：

1. 流量数据保存
  - 生成 inflow.npy 和 outflow.npy
  - 维度：(n, t)，其中 n=站点数，t=时间序列长度

2. 训练数据切片

  - 输出文件至 datasets/your_dataset/npy_data/
  - 命名格式：inflow_{index}.npy 和 outflow_{index}.npy
  - 维度：(n, t'+1)，其中 t'=输入时间步数

3. 生成 JSONL 索引
  - 输出文件至 datasets/your_dataset_training/

```json
{"inflow_path": "datasets/your_dataset/npy_data/inflow_0.npy", "outflow_path": "datasets/your_dataset/npy_data/outflow_0.npy"}
......
```
### ⚙️ 模型训练
修改配置文件 `config/st_llm_attn.yml` 后运行：
```bash
python train.py --model_config_path config/st_llm_attn.yml
```
### 📊 模型评估
你可以运行以下脚本开始评估模型，评价指标包括RMSE、MAE 、WMAPE
```bash
# 评估前需确保模型已训练保存至 saved_model/
python example/eval.py  --model_config_path config/st_llm_attn.yml --data_path gaotie_trainning/test.jsonl --model_path saved_model/st_llm_attn
```

### 🔮 模型推理
你可以运行以下脚本开始进行模型推理，这依赖于你之前生成的inflow.npy和outflow.npy
```bash
# 需提前准备好 inflow.npy 和 outflow.npy
python example/inference.py \
  --model_config_path config/st_llm_attn.yml \
  --inflow_path datasets/your_dataset/inflow.npy \
  --outflow_path datasets/your_dataset/outflow.npy \
  --model_path saved_model/st_llm_attn \
  --step 7 \                  # 预测步长
  --save_dir results/         # 输出目录
```
### 📈 结果可视化
首先你需要进行模型推理，然后运行以下脚本即可进行结果可视化
```bash
python example/plot.py --save_dir results/
```