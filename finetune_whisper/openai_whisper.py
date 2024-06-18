import torch
from torch import nn
import warnings
import numpy as np
import whisper  # type: ignore
import evaluate  # type: ignore
from pytorch_lightning import LightningModule, Trainer
from transformers import get_linear_schedule_with_warmup  # type: ignore
from torch.optim import AdamW
from torch.utils.data import IterableDataset, Dataset, DataLoader
from whisper import load_model  # type: ignore
from typing import Union, Optional, Any, Callable
from pathlib import Path
import whisper.tokenizer  # type: ignore
from dataclasses import dataclass, asdict
from collections import OrderedDict
from glob import glob

# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


from .training_proto import GenericSample
from .utils import prepare_dataset


@dataclass
class Config:
    output_dir: Union[str, Path]
    learning_rate: float
    weight_decay: float
    adam_epsilon: float
    warmup_steps: int
    batch_size: int
    num_workers: int
    num_train_epochs: int
    gradient_accumulation_steps: int


class WhisperModelModule(LightningModule):
    def __init__(
        self,
        config: Config,
        model: whisper.Whisper,
        tokenizer: whisper.tokenizer.Tokenizer,
        options: whisper.DecodingOptions,
    ) -> None:
        super().__init__()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.special_tokens = set(self.tokenizer.special_tokens.values())  # hashset

        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

    def forward(self, x):
        return self.model(x)

    def decode_without_special_tokens(self, tokens):
        return self.tokenizer.decode(
            [token for token in tokens if token not in self.special_tokens]
        )

    def training_step(self, batch, _):
        if batch is None:
            # skip batches that are None
            return None

        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, _):
        if batch is None:
            # skip batches that are None
            return None

        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.decode_without_special_tokens(o))
            l_list.append(self.decode_without_special_tokens(l))

        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {"cer": cer, "wer": wer, "loss": loss}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.t_total,
        )
        self.scheduler = scheduler

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def setup_training_dataset_length(self, length: int) -> None:
        self.t_total = (
            (length // (self.config.batch_size))
            // self.config.gradient_accumulation_steps
            * float(self.config.num_train_epochs)
        )

    def setup(self, stage=None):

        # if stage == "fit" or stage is None:
        #     self.t_total = (
        #         (len(self.__train_dataset) // (self.config.batch_size))
        #         // self.config.gradient_accumulation_steps
        #         * float(self.config.num_train_epochs)
        #     )
        pass


@dataclass
class OpenAIWhisperSpecificSample:
    input_ids: Any
    labels: Any
    dec_input_ids: Any


class WhisperDataCollatorWhithPadding:
    def __call__(self, features: Optional[list[Optional[OpenAIWhisperSpecificSample]]]):
        if features is None:
            return None

        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            if f is None:
                # if one of the items in the batch is None discard the whole batch
                return None
            input_ids.append(f.input_ids)
            labels.append(f.labels)
            dec_input_ids.append(f.dec_input_ids)

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])  # type: ignore # TODO

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # TODO: Here we are padding labels and dec_input_ids to make the whole batch
        # have the same length (but why ????)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len), "constant", constant_values=-100)
            for lab, lab_len in zip(labels, label_lengths)
        ]

        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), "constant", constant_values=50257)
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ]  # 50257 is eot token id

        batch = {"labels": labels, "dec_input_ids": dec_input_ids}

        batch = {
            k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()  # type: ignore # TODO
        }
        batch["input_ids"] = input_ids

        return batch


class OpenAIWhisperTrainingStrategy:
    """
    OpenAI whisper training strategy uses `openai-whisper` with `ppytorch_lightning` to finetune a given whisper checkpoint.


    Args:
        name (`str`, `pathlib.Path`): A hardcoded name of the model or a path to a checkpoint.
            Currently those are the hardcoded-model names:
            `tiny.en`, `tiny`, `base.en`, `base`, `small.en`, `small`, `medium.en`, `medium`, `large-v1`, `large-v2`, `large-v3`, `large`

        config (`Config`): the training parameters.

        train_id (`str`): the train id used for tensorboard logger.

        train_name (`str`): the train name used for tensorboard logger.

        language (`str`): the language of the tokenizer.

        device (`str`, *optional*, defaults to `None`):
            the whisper model will be loaded on that device, defaults to `cuda` if available otherwise `cpu`.

        without_timestamps (`bool`, *optional*, defaults to `True`):
            whether training should consider timestamps or not.

        skip_exceeding_30_secs (`bool`, *optional*, defaults to `True`):
            option to determine the expected behavior of samples exceeding 30-sec limit of whisper
            (skip that sample or trim it).
    """

    def __init__(
        self,
        name: Union[str, Path],
        *,
        config: Config,
        train_id: str,
        train_name: str,
        language: str,
        device: Optional[str] = None,
        without_timestamps: bool = True,
        skip_exceeding_30_secs: bool = True,
    ):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.skip_exceeding_30_secs = skip_exceeding_30_secs

        self.allowed_special = set() if without_timestamps else "all"

        self.device = device
        self.model = load_model(name, device)

        self.dims = asdict(self.model.dims)

        self.options = whisper.DecodingOptions(
            language=language, without_timestamps=without_timestamps
        )
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language=language, task=self.options.task
        )
        self.config = config
        self.training_ready_model = WhisperModelModule(
            self.config,
            self.model,
            self.tokenizer,
            self.options,
        )

        self.train_id = train_id
        self.train_name = train_name

        def default_general_error_fn(_: GenericSample, e: Optional[Exception]) -> None:
            warnings.warn(str(e))

        self.exceeding_30_secs_fn = lambda _: None
        self.general_error_fn = default_general_error_fn

    def train(
        self,
        training_dataset: Union[Dataset[GenericSample], IterableDataset[GenericSample]],
        validation_dataset: Union[
            Dataset[GenericSample], IterableDataset[GenericSample]
        ],
    ) -> None:
        """
        Start the training process.

        Args:
            training_dataset (`Dataset[GenericSample]`, `IterableDataset[GenericSample]`):
                the training dataset of `GenericSample` samples.

            validation_dataset (`Dataset[GenericSample]`, `IterableDataset[GenericSample]`):
                the validation dataset of `GenericSample` samples.
        """

        self.train_dataloader = DataLoader(
            prepare_dataset(
                training_dataset, map_to=self.map_to_openai_whisper_specific_sample
            ),
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,  # type: ignore
            drop_last=True,
            collate_fn=WhisperDataCollatorWhithPadding(),
        )

        self.val_dataloader = DataLoader(
            prepare_dataset(
                validation_dataset, map_to=self.map_to_openai_whisper_specific_sample
            ),
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,  # type: ignore
            collate_fn=WhisperDataCollatorWhithPadding(),
        )

        # for batch in self.train_dataloader:
        #     for dec_input_ids in batch["dec_input_ids"]:
        #         # print(f"{len(dec_input_ids) = }, {dec_input_ids}")
        #         print(
        #             self.training_ready_model.decode_without_special_tokens(
        #                 dec_input_ids
        #             )
        #         )

        #     print("============")

        log_dir = Path(self.config.output_dir) / "logs"
        checkpoint_dir = Path(self.config.output_dir) / "artifacts"
        log_dir.mkdir()
        checkpoint_dir.mkdir()

        # tflogger = TensorBoardLogger(
        #     save_dir=log_dir, name=self.train_name, version=self.train_id
        # )

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{checkpoint_dir}/checkpoint",
            filename="checkpoint-{epoch:04d}",
            save_top_k=-1,  # all model save
        )

        callback_list = [
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ]

        trainer = Trainer(
            precision="16-mixed",
            accelerator=self.device,
            max_epochs=self.config.num_train_epochs,
            accumulate_grad_batches=self.config.gradient_accumulation_steps,
            # logger=tflogger,
            logger=None,
            callbacks=callback_list,
        )

        self.training_ready_model.setup_training_dataset_length(len(training_dataset))  # type: ignore

        trainer.fit(
            self.training_ready_model, self.train_dataloader, self.val_dataloader
        )

        last_checkpoint_path = sorted(glob(f"{checkpoint_dir}/checkpoint/*"))[-1]

        state_dict = torch.load(last_checkpoint_path)

        # state_dict = self.training_ready_model.state_dict()

        compatible_state_dict: dict[str, Any] = {}
        compatible_state_dict["model_state_dict"] = OrderedDict()
        compatible_state_dict["dims"] = self.dims

        for key, value in state_dict["state_dict"].items():
            # for key, value in state_dict.items():
            if key.startswith("model."):
                key = key[6:]
            compatible_state_dict["model_state_dict"][key] = value

        torch.save(
            compatible_state_dict,
            f"{checkpoint_dir}/checkpoint/whisper_compatible_last_checkpoint.ckpt",
        )

    def on_audio_exceeding_30_secs(self, fn: Callable[[GenericSample], None]):
        """
        Setup a callback for audio waveforms that exceed 30-sec limit.

        Args:
            fn (`Callable`): A function that will be called with `GenericSample` samples exceeding 30 seconds.
        """
        self.exceeding_30_secs_fn = fn

    def on_general_error(
        self, fn: Callable[[GenericSample, Optional[Exception]], None]
    ):
        """
        Setup a callback for problematic audio waveforms.

        Args:
            fn (`Callable`): A function that will be called with the problematic `GenericSample` sample
            and an *optional* Exception that was raised (it could be None).
        """
        self.general_error_fn = fn  # type: ignore

    def map_to_openai_whisper_specific_sample(
        self,
        sample: GenericSample,
    ) -> Optional[OpenAIWhisperSpecificSample]:
        try:
            if sample.waveform is None or sample.text is None:
                self.general_error_fn(sample, None)
                return None

            # TODO: make sure to cover all cases (ex: mono waveform)
            num_channels = sample.waveform.shape[0]

            if num_channels > 1:
                audio = sample.waveform[0]
            else:
                audio = sample.waveform

            flattened_audio = audio.flatten()

            if flattened_audio.shape[-1] > whisper.audio.N_SAMPLES:
                self.exceeding_30_secs_fn(sample)
                if self.skip_exceeding_30_secs:
                    return None

            audio = whisper.pad_or_trim(flattened_audio)
            mel = whisper.log_mel_spectrogram(audio)

            text = [*self.tokenizer.sot_sequence] + self.tokenizer.encode(
                sample.text, allowed_special=self.allowed_special
            )

            labels = text[1:] + [self.tokenizer.eot]

            return OpenAIWhisperSpecificSample(
                input_ids=mel, labels=labels, dec_input_ids=text
            )
        except Exception as e:
            self.general_error_fn(sample, e)
            return None
