import whisper
import torch

print(torch.cuda.is_available())
model = whisper.load_model("large-v2")
# add translate option
# options = whisper.DecodingOptions(language="ko", task="translate")
result = model.transcribe("./../data/output.wav", language="ko", task="translate")
print(result["text"])
