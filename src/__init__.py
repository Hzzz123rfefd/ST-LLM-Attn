from src.model import STLLMAttn,STLLMAttnLSTM, STLLMAttnNoAttn, STLLMAttnNoLLM, STLLMAttnDecoder, BPNN, CNN, GRU, LSTM, STResNet, Transformer, WaveNet, TCN
from src.dataset import DatasetForTimeSeries


datasets = {
    "time_series": DatasetForTimeSeries,
}

models = {
    "st_llm_attn":STLLMAttn,
    "st_llm_attn_lstm":STLLMAttnLSTM,
    "st_llm_attn_decoder":STLLMAttnDecoder,
    "st_llm_attn_no_attn":STLLMAttnNoAttn,
    "st_llm_attn_no_llm":STLLMAttnNoLLM,
    "bpnn": BPNN,
    "cnn":CNN,
    "gru":GRU,
    "lstm":LSTM,
    "st_resnet": STResNet,
    "transformer": Transformer,
    "wave_net": WaveNet,
    "tcn": TCN
}
