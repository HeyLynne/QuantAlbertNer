# QuantAlbertNer
Quantify Albert for NER(Named Entity Recognition) also implement Albert with focal loss

This repository contains an implementation of [Albert](https://arxiv.org/pdf/1909.11942.pdf) in Pytorch for NER task. The code was based on the [albert_pytorch](https://github.com/lonePatient/albert_pytorch).

## Dependicies
Please check out the requirements.txt. For installation:
```
pip install -r requirements.txt
```

## Pretrained model
We have already download the albert chinese tiny in prev_trained_model. If you need more pre-trained model please checkout:
### Albert Chinese Pre-trained Model Download
- albert_tiny_zh:https://drive.google.com/open?id=1qAykqB2OXIYXSVMQSt_EDEPCorTAkIvu
- albert_small_zh:https://drive.google.com/open?id=1t-DJKAqALgwlO8J_PZ3ZtOy0NwLipGxA
- albert_base_zh:https://drive.google.com/open?id=1m_tnylngzEA94xVvBCc3I3DRQNbK32z3
- albert_large_zh:https://drive.google.com/open?id=19UZPHfKJZY9BGS4mghuKcFyAIF3nJdlX
- albert_xlarge_zh:https://drive.google.com/open?id=1DdZ3-AXaom10nrx8C99UhFYsx1BYYEJx
- albert_xxlarge_zh:https://drive.google.com/open?id=1F-Mu9yWvj1XX5WN6gtyxbVLWr610ttgC

## Fine-tune
You can fine-tune model with Albert and Albert-Focal loss.
```

```

## Quantization
We implemented the dynamic quantization version of Albert. If you need to quantify albert, you need to do with the following instructions:
1. Fine-tune the albert in your own dataset.
2. Load your own model and run the run_quantalbert_ner.py

## Dataset
The original dataset is as follows:
```
也/B-O 许/B-O 放/B-Verb 弃/I-Verb 才/B-O 能/B-O 靠/B-O 近/B-O 你/B-O
```
The format of label is as follows:
```
其他B B-O
其他I I-O
动作B B-Verb
动作I I-Verb
```

## Result
We quantified albert in our own dataset. The inference speed increased 10% but the accuracy drops 3%. And the performance was 0.5% higher when we use focal loss. But the time cose increase 30%. 

## Todo
- Static quantify the model.
- Run the model in an open-source dataset.

If you have any questions please contact me at lion19930924@163.mail. Thanks a lot.
