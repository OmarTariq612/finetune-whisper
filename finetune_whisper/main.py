import sys
import time

import torch.utils
from torch.utils.data import (
    IterableDataset,
    Dataset,
    Sampler,
    SequentialSampler,
    RandomSampler,
)

from whisper import load_model  # type: ignore

import torch.utils.data
import torchaudio  # type: ignore
import torchaudio.transforms as at  # type: ignore
import numpy as np
import sys
import torch
from typing import (
    Protocol,
    Iterator,
    Iterable,
    Union,
    Optional,
    Callable,
    TypeVar,
    Collection,
    Sequence,
    Generic,
    Any,
)

from pathlib import Path
import warnings

from dataclasses import dataclass

import whisper.tokenizer  # type: ignore

from .training_proto import GenericSample
from .utils import prepare_dataset, dataset_from_sequence
from .openai_whisper import (
    OpenAIWhisperTrainingStrategy,
    Config,
)


def load_wave(wave_path, sample_rate=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True, format="mp3")
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


# class QuranDataset(torch.utils.data.Dataset):
#     def __init__(self, audio_info_list, tokenizer, sample_rate):
#         super().__init__()
#         self.audio_info_list = audio_info_list
#         self.sample_rate = sample_rate
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.audio_info_list)

#     def __getitem__(self, index):
#         info = self.audio_info_list[index]
#         audio_path = info[0]
#         text = info[1]

#         # audio
#         try:
#             audio = load_wave(audio_path, sample_rate=self.sample_rate)[
#                 0
#             ]  # indexing one channel before flattening
#             audio = whisper.pad_or_trim(audio.flatten())
#             mel = whisper.log_mel_spectrogram(audio)

#             text = [*self.tokenizer.sot_sequence] + self.tokenizer.encode(
#                 text, allowed_special="all"
#             )
#             labels = text[1:] + [self.tokenizer.eot]

#             return {"input_ids": mel, "labels": labels, "dec_input_ids": text}
#         except Exception as e:
#             warnings.warn(e.str())
#             return None


# class WhisperDataCollatorWhithPadding:
#     def __call__(self, features):
#         if features is None:
#             return None

#         input_ids, labels, dec_input_ids = [], [], []
#         for f in features:
#             if f is None:
#                 return None
#             input_ids.append(f["input_ids"])
#             labels.append(f["labels"])
#             dec_input_ids.append(f["dec_input_ids"])

#         input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

#         label_lengths = [len(lab) for lab in labels]
#         dec_input_ids_length = [len(e) for e in dec_input_ids]
#         max_label_len = max(label_lengths + dec_input_ids_length)

#         labels = [
#             np.pad(lab, (0, max_label_len - lab_len), "constant", constant_values=-100)
#             for lab, lab_len in zip(labels, label_lengths)
#         ]
#         dec_input_ids = [
#             np.pad(e, (0, max_label_len - e_len), "constant", constant_values=50257)
#             for e, e_len in zip(dec_input_ids, dec_input_ids_length)
#         ]  # 50257 is eot token id

#         batch = {"labels": labels, "dec_input_ids": dec_input_ids}

#         batch = {
#             k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()
#         }
#         batch["input_ids"] = input_ids

#         return batch


def main() -> None:
    # tokenizer = whisper.tokenizer.get_tokenizer(
    #     True,
    #     language="ar",
    #     task=whisper.DecodingOptions(language="ar", without_timestamps=True).task,
    # )

    # training_dataset = QuranDataset(
    #     [
    #         ("../1.mp3", "أولئك على هدى من ربهم وأولئك هم المفلحون"),
    #         (
    #             "../2.mp3",
    #             "إن الذين كفروا سواء عليهم أأنذرتهم أم لم تنذرهم لا يؤمنون",
    #         ),
    #         (
    #             "../3.mp3",
    #             "ختم الله على قلوبهم وعلى سمعهم وعلى أبصارهم غشاوة ولهم عذاب عظيم",
    #         ),
    #         (
    #             "../4.mp3",
    #             "ومن الناس من يقول آمنا بالله وباليوم الآخر وما هم بمؤمنين",
    #         ),
    #         (
    #             "../5.mp3",
    #             "يخادعون الله والذين آمنوا وما يخدعون إلا أنفسهم وما يشعرون",
    #         ),
    #     ],
    #     tokenizer=tokenizer,
    #     sample_rate=16000,
    # )

    # def decode_without_special_tokens(tokenizer):
    #     special_tokens = set(tokenizer.special_tokens.values())  # hashset

    #     def decode_without_special(tokens):
    #         return tokenizer.decode(
    #             [token for token in tokens if token not in special_tokens]
    #         )

    #     return decode_without_special

    # training_dataloader = torch.utils.data.DataLoader(
    #     training_dataset,
    #     batch_size=2,
    #     num_workers=1,
    #     collate_fn=WhisperDataCollatorWhithPadding(),
    # )

    # decode_fn = decode_without_special_tokens(tokenizer=tokenizer)

    # for batch in training_dataloader:
    #     for dec_input_ids in batch["dec_input_ids"]:
    #         print(decode_fn(dec_input_ids))
    #     print("====================")

    def from_path_to_waveform(path_and_text: tuple[str, str]) -> GenericSample:

        try:
            path = path_and_text[0]
            text = path_and_text[1]

            waveform, sample_rate = torchaudio.load(path)
            if sample_rate != 16000:
                waveform = at.Resample(sample_rate, 16000)(waveform)

            return GenericSample(waveform=waveform, text=text, context=None)
        except:
            return GenericSample(waveform=None, text=text, context=None)

    trainin_dataset = prepare_dataset(
        dataset_from_sequence(
            [
                ("../1.mp3", "أولئك على هدى من ربهم وأولئك هم المفلحون"),
                (
                    "../2.mp3",
                    "إن الذين كفروا سواء عليهم أأنذرتهم أم لم تنذرهم لا يؤمنون",
                ),
                (
                    "../3.mp3",
                    "ختم الله على قلوبهم وعلى سمعهم وعلى أبصارهم غشاوة ولهم عذاب عظيم",
                ),
                (
                    "../4.mp3",
                    "ومن الناس من يقول آمنا بالله وباليوم الآخر وما هم بمؤمنين",
                ),
                (
                    "../5.mp3",
                    "يخادعون الله والذين آمنوا وما يخدعون إلا أنفسهم وما يشعرون",
                ),
                # (
                #     "../5.mp3",
                #     "يخادعون الله والذين آمنوا وما يخدعون إلا أنفسهم وما يشعرون",
                # ),
                # (
                #     "../5.mp3",
                #     "يخادعون الله والذين آمنوا وما يخدعون إلا أنفسهم وما يشعرون",
                # ),
                # (
                #     "../5.mp3",
                #     "يخادعون الله والذين آمنوا وما يخدعون إلا أنفسهم وما يشعرون",
                # ),
                # (
                #     "../5.mp3",
                #     "يخادعون الله والذين آمنوا وما يخدعون إلا أنفسهم وما يشعرون",
                # ),
                # (
                #     "../5.mp3",
                #     "يخادعون الله والذين آمنوا وما يخدعون إلا أنفسهم وما يشعرون",
                # ),
                # (
                #     "../5.mp3",
                #     "يخادعون الله والذين آمنوا وما يخدعون إلا أنفسهم وما يشعرون",
                # ),
            ]
        ),
        shuffle=False,
        map_to=from_path_to_waveform,
    )

    validation_dataset = dataset_from_sequence([])

    trainer = OpenAIWhisperTrainingStrategy(
        "tiny",
        config=Config(
            output_dir="here",
            learning_rate=0.005,
            weight_decay=0.01,
            adam_epsilon=1e-8,
            warmup_steps=0,
            batch_size=1,
            num_workers=1,
            num_train_epochs=1,
            gradient_accumulation_steps=1,
        ),
        language="ar",
        train_id="01",
        train_name="whisper",
    )  # type: ignore

    trainer.train(
        training_dataset=trainin_dataset, validation_dataset=validation_dataset
    )

    # from glob import glob

    # print(sorted(glob("here/artifacts/checkpoint/*"))[-1])

    # model = whisper.load_model(
    #     "/home/omar/programming/graduation-project/finetune-whisper/finetune_whisper/here/artifacts/checkpoint/whisper_compatible_last_checkpoint.ckpt"
    # )

    # model = whisper.load_model("tiny")

    # print(model.transcribe("../5.mp3", language="ar"))


if __name__ == "__main__":
    main()
