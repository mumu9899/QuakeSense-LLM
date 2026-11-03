QuakeSense-LLM
Overview
QuakeSense-LLM is an inference framework for classifying perceived shaking in Chinese post-earthquake social media text. Given a short text such as a Weibo comment, the model predicts one of four perceived-shaking classes.

The model reported in the related paper uses DeepSeek-7B-Base with task-specific LoRA fine-tuning.

These classes describe perceived shaking expressed in social media text and are not equivalent to official seismic-intensity levels.

This repository focuses on the text-classification component of the study.

Installation
1. Clone the repository
Replace <repository-url> with the actual repository URL.

SHELL
复制
git clone <repository-url>
cd QuakeSense-LLM
2. Install dependencies
Python 3.10+ is recommended. A GPU is recommended for large language model inference.

SHELL
复制
pip install -r requirements.txt
3. Prepare the model files
Place the released model files in the pretrained/ directory.

TEXT
复制
pretrained/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── label_mapping.json
The model reported in the paper uses DeepSeek-7B-Base as the pretrained backbone, with LoRA applied to q_proj, k_proj, v_proj, and o_proj. If the released checkpoint contains only the LoRA adapter, the corresponding DeepSeek-7B-Base backbone is also required.

Quick Start
Single-text prediction
SHELL
复制
python predict.py --input "刚才床晃了一下，好像地震了"
Example output:

TEXT
复制
Input text: 刚才床晃了一下，好像地震了
Predicted label: 1
Predicted category: Weak perceived shaking
Confidence: 0.8734
Batch prediction
SHELL
复制
python predict.py --input sample_input.csv --output predictions.csv
The input CSV should contain a column named text.

CSV
复制
text
刚才床晃了一下，好像地震了
灯晃得很明显，大家都跑出来了
墙裂了，东西掉了一地
The output CSV contains the original text, predicted label, predicted category, confidence score, and class probabilities.

TEXT
复制
text,pred_label,pred_name,confidence,prob_0,prob_1,prob_2,prob_3
Label Definitions
The implementation uses integer labels 0-3, corresponding to Classes 0, I, II, and III in the paper.

Label	Paper class	Category	Description
0	Class 0	Irrelevant/non-perceived shaking	No personal shaking experience is described, such as prayers, reposted news, event discussions, earthquake-knowledge comments, or other content without perceived-shaking information.
1	Class I	Weak perceived shaking	Weak or uncertain shaking, slight object motion, short-duration shaking, or other low-level shaking cues.
2	Class II	Moderate perceived shaking	Clearly noticeable shaking, moderate object responses, longer-duration shaking, mild damage, or relatively strong emotional reactions.
3	Class III	Strong perceived shaking	Strong or violent shaking, severe object responses, substantial damage, strong physical reactions, or alarm/panic.
The annotation scheme considers six types of cues: bodily sensation, object motion, shaking duration, building/object damage, degree adverbs, and emotional expressions. Representative expressions guide annotation rather than serving as deterministic keyword rules. When multiple levels occur in the same text, the highest expressed shaking level is used as the final label.

Model
Item	Value
Base model	DeepSeek-7B-Base
Fine-tuning method	Task-specific LoRA fine-tuning
LoRA target modules	q_proj, k_proj, v_proj, o_proj
Classification head	Linear + Softmax
Max sequence length	256 tokens
Number of labels	4
Training objective	Multiclass cross-entropy loss
Reported Performance
The model was evaluated using an event-independent strategy. Samples from 11 earthquakes were used for training and validation, while 12,868 samples from the Luding earthquake were used as an independent test set.

Metric	Result
Accuracy	91.34%
Macro-average F1-score	89.78%
Weighted-average F1-score	91.40%
Across five leave-one-event-out tests, the macro-average F1-score ranged from 0.898 to 0.980.

Data Availability
The study contains 64,731 valid annotated Weibo comments from 12 earthquakes.

The annotated dataset is intended for public release upon publication of the related article. User-identifying information has been removed. Before public release, the data are available from the corresponding author upon reasonable request.

Limitations
Data were collected from comment sections under the official earthquake-reporting Weibo account of the China Earthquake Networks Center (CENC).
The sample distribution may be influenced by platform user composition, commenting activity, and the reach of the official account.
Weibo comments are colloquial, fragmented, and may contain irrelevant or ambiguous expressions.
The model classifies perceived-shaking expressions in text and should not be interpreted as estimating official seismic-intensity levels.
Further validation on additional earthquake events is needed.
Repository Files
model.py — Model loading and inference.
predict.py — Single-text and batch prediction.
pretrained/ — Model and tokenizer files.
requirements.txt — Python dependencies.
sample_input.csv — Example batch input.
LICENSE — Open-source license.
Related Paper
Classifying Perceived Shaking in Post-Earthquake Social Media with Large Language Models for Rapid Estimation of Seismic Intensity-Affected Areas

Qingle Cheng, Zhengxuan Song, Maosen Yang, Linlin Xie

If you use this repository in academic work, please cite the related paper. Bibliographic information should be updated after formal publication.

BIBTEX
复制
@misc{cheng2026perceivedshaking,
  title  = {Classifying Perceived Shaking in Post-Earthquake Social Media with Large Language Models for Rapid Estimation of Seismic Intensity-Affected Areas},
  author = {Cheng, Qingle and Song, Zhengxuan and Yang, Maosen and Xie, Linlin},
  year   = {2026},
  note   = {Manuscript; update bibliographic information after formal publication}
}
Contact
For questions about this repository, please contact:

Qingle Cheng
Email: chengql94@163.com
