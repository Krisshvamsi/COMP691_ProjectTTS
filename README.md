---
language: "en"
tags:
- text-to-speech
- TTS
- speech-synthesis
- Tacotron2
- speechbrain
license: "apache-2.0"
datasets:
- LJSpeech
metrics:
- mos
pipeline_tag: text-to-speech
---


# Text-to-Speech (TTS) with Transformer trained on LJSpeech

This repository provides all the necessary tools for Text-to-Speech (TTS)  with SpeechBrain using a [Transformer](https://arxiv.org/pdf/1809.08895.pdf) pretrained on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

The pre-trained model takes in input a short text and produces a spectrogram in output. One can get the final waveform by applying a vocoder (e.g., HiFIGAN) on top of the generated spectrogram.


## Install SpeechBrain

```
pip install speechbrain
```
### Perform Text-to-Speech (TTS) - Running Inference
To run model inference pull the interface directory as shown in the cell below

Note: Run on T4-GPU for faster inference
```
!pip install --upgrade --no-cache-dir gdown
!gdown 1oy8Y5zwkLel7diA63GNCD-6cfoBV4tq7
!unzip inference.zip
```
```python
%%capture
!pip install speechbrain
%cd inference
```

```python
import torchaudio
from TTSModel import TTSModel
from IPython.display import Audio
from speechbrain.inference.vocoders import HIFIGAN

texts = ["This is a sample text for synthesis."]

model_source_path = "/content/inference"
# Intialize TTS (Transformer) and Vocoder (HiFIGAN)
my_tts_model = TTSModel.from_hparams(source=model_source_path)
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

# Running the TTS
mel_output = my_tts_model.encode_text(texts)

# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output)

# Save the waverform
torchaudio.save('example_TTS.wav',waveforms.squeeze(1), 22050)
print("Saved the audio file!")
```

If you want to generate multiple sentences in one-shot, pass the sentences as items in a list.


### Inference on GPU
To perform inference on the GPU, add  `run_opts={"device":"cuda"}`  when calling the `from_hparams` method.

Note: For Training the model please visit this TTS_Training_Inference notebook

For the inference API, please visit the huggingface interface at (https://huggingface.co/Krisshvamsi/TTS)

### Limitations
The SpeechBrain team does not provide any warranty on the performance achieved by this model when used on other datasets.

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
