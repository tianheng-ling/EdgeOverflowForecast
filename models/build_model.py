from .fp32.RNN.FloatRNN import FloatRNN
from .fp32.Transformer import FloatTransformer
from .quant.RNN.QuantRNN import QuantRNN
from .quant.Transformer import QuantTransformer


def build_model(model_params: dict):

    is_qat = model_params.get("is_qat", False)
    model_type = model_params.get("model_type")

    model_mapping = {
        "lstm": (QuantRNN, FloatRNN),
        "transformer": (QuantTransformer, FloatTransformer),
    }

    if model_type not in model_mapping:
        raise ValueError(f"Model Type '{model_type}' is not recognized")

    ModelClass = (
        model_mapping[model_type][0] if is_qat else model_mapping[model_type][1]
    )

    return ModelClass(**model_params)
