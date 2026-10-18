import math
import torch

def answer(question, logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be finite and positive")
    if logits.ndim != 1 or len(logits) != len(question.keys) or not torch.isfinite(logits).all():
        raise ValueError("Expected one finite logit per candidate")
    probs = (logits.float() / temperature).softmax(-1).detach().cpu().tolist()
    total = sum(probs)
    probs = [p / total for p in probs]
    result = {"type": question.kind}
    if question.kind == "noul":
        result["noul"] = probs[question.keys.index("true")]
        return result
    result["probabilities"] = dict(zip(question.keys, probs))
    mode = max(range(len(probs)), key=probs.__getitem__)
    if question.kind == "choice":
        result["choice"] = question.keys[mode]
        result["confidence"] = 1.0 if len(probs) == 1 else max(0.0, (max(probs) - 1/len(probs)) / (1 - 1/len(probs)))
    else:
        result["score"] = sum(i * p for i, p in enumerate(probs))
        result["legend"] = dict(zip(question.keys, question.descriptions))
        # Independently implemented formula from pinned system-one adapter:
        # fb52b1030b7fc1f4f1cf39910afa5da54f9835e3, confidence_metrics.py.
        distance = sum(p * abs(i - mode) for i, p in enumerate(probs))
        uniform_mad = sum(abs(i - (len(probs)-1)/2) for i in range(len(probs))) / len(probs)
        result["confidence"] = max(0.0, 1.0 - distance / uniform_mad)
    return result
