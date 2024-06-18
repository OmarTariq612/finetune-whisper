import torch.utils
from torch.utils.data import (
    IterableDataset,
    Dataset,
)

import torch.utils.data
import torch
from typing import (
    Protocol,
    Union,
    Optional,
    Any,
)
from dataclasses import dataclass, field


@dataclass
class GenericSample:
    """
    GenericSample contains a sample that has a waveform, a text that can be provided as a dataset for a `TrainingStrategy`
    and a context the can be used for error handling.

    Args:
        waveform (`torch.Tensor`, *optional*, defaults to `None`):
            the waveform of the sample (with sample_rate = 16000Hz), [channel, time] (channels are first).

        text: (`str`, *optional*, defaults to `None`):
            the coresponding text of the sample.

        context: (`Any`, *optional*, defaults to `None`):
            a context that you can add to add more info about the sample for error handling purposes.
    """

    waveform: Optional[torch.Tensor] = field(default=None)
    text: Optional[str] = field(default=None)
    context: Any = field(default=None)


class TrainingStrategy(Protocol):
    def train(
        self,
        training_dataset: Union[Dataset[GenericSample], IterableDataset[GenericSample]],
        validation_dataset: Union[
            Dataset[GenericSample], IterableDataset[GenericSample]
        ],
    ) -> None: ...


# def resample_to(waveform, current_sample_rate, desired_sample_rate):
#     """
#     Resample the given waveform to the `desired_sample_rate`
#     """
#     desired_waveform = at.Resample(current_sample_rate, desired_sample_rate)(waveform)
#     return desired_waveform


# def load_wave_mono(
#     wave_path, resample=True, sample_rate=16000, format="mp3", backend=None
# ):
#     """
#     load a waveform given the `wave_path`, resampling is optional (true by default)
#     """
#     waveform, sr = torchaudio.load(
#         wave_path, normalize=False, format=format, backend=backend
#     )
#     if sample_rate != sr and resample:
#         waveform = resample_to(waveform, sr, sample_rate)
#         sr = sample_rate
#     return waveform[0].flatten().numpy(), sr


# def save_wave(waveform, output_path, sample_rate, backend=None):
#     """
#     save the given waveform as .wav file
#     """
#     return torchaudio.save(
#         output_path,
#         torch.tensor(np.array([waveform])),
#         sample_rate=sample_rate,
#         backend=backend,
#     )


# class DatasetFromCollection(Dataset):
#     def __init__(self, collection: Collection):
#         self.collection = collection

#     def __len__(self):
#         return len(self.collection)

#     def __getitem__(self, index):
#         return self.collection[index]


def main() -> None:
    # torch.tensor()
    # if len(sys.argv) != 3:
    #     print(f"{sys.argv[0]} input_filepath output_filepath")
    #     exit(1)

    # input_filepath = sys.argv[1]
    # output_filepath = sys.argv[2]

    # # torchaudio.set_audio_backend("sox_io")
    # wave, sample_rate = load_wave_mono(input_filepath, backend="ffmpeg")

    # save_wave(wave, output_filepath, sample_rate, backend="ffmpeg")

    # d = [(1, 2, "omar"), (2, 3, "test")]
    # dataset: Dataset = d

    # d = MyOwnDataset([1, 2, 3, 4, 5])
    # dataloader = torch.utils.data.DataLoader(d, batch_size=2, shuffle=True)

    # dd = MyOwnIterableDataset([1, 2, 3, 4, 5])
    # dataloader = torch.utils.data.DataLoader(dd, batch_size=2, shuffle=True)

    # for item in dataloader:
    #     print(item)

    # print(d[5])
    # print(d[5, 2, 3, 4, 20])

    # i: Iterable[int] = MyIterable(1, 5)
    # print(i.__getitem__(3))

    # d = MyOwnIterableDataset([1, 5, 2, 3, 15])

    # for i in MyIterable(1, 5):
    #     print(i)

    # l: Union[list[str], list[Path]] = ["omar", Path("test")]

    # d = MyOwnIterableDataset[str](
    #     dataset=dataset_from_dict({0: "Omar", 1: "Reem", 2: "Tariq"}),
    #     sampler=RandomSampler([1, 2, 3]),
    # )

    # for dd in d:
    #     print(dd)

    # print(d[2])

    # training_dataset = dataset(
    #     dataset_from_sequence([filepath for filepath in glob("*")]),
    #     shuffle=True,
    #     map_to=lambda elem: elem,
    # )

    # training_strategy = OpenAIWhisperTrainingStrategy(
    #     "small", device="cuda"
    # ).configure()

    # training_strategy.setup_datasets(training_dataset=training_dataset)

    # training_strategy.train()

    # d = dataset(
    #     iterable_dataset_from_collection([1, 5, 2, 4, 3, 2]),
    #     shuffle=False,
    #     map_to=lambda elem: elem,
    # )

    # for i in torch.utils.data.DataLoader(
    #     d, batch_size=2, collate_fn=lambda e: (e[1], e[0])
    # ):
    #     print(i)

    # import whisper.tokenizer
    # tokenizer = whisper.tokenizer.get_tokenizer(True, language="ar", task="transcribe")
    # print(tokenizer.special_tokens.values().mapping)

    # d: dict[int, int] = {}
    # s: set[int] = set()

    # for i in range(10_000_000):
    #     d[i] = i + 1
    #     s.add(i + 1)

    # needle = int(sys.argv[1])

    # start = time.monotonic()
    # print(needle in d.values())
    # duration_1 = time.monotonic() - start

    # start = time.monotonic()
    # print(needle in s)
    # duration_2 = time.monotonic() - start

    # print(f"{duration_1:.10f}\n{duration_2:.10f}")

    # def from_path_to_waveform(path_and_text: tuple[str, str, int]) -> GenericSample:
    #     path = path_and_text[0]
    #     text = path_and_text[1]

    #     waveform, sample_rate = torchaudio.load(path)
    #     if sample_rate != 16000:
    #         waveform = at.Resample(sample_rate, 16000)(waveform)

    #     return GenericSample(waveform=waveform, text=text, number=path_and_text[2])

    # trainin_dataset = dataset(
    #     dataset_from_sequence(
    #         [
    #             ("../1.mp3", "أولئك على هدى من ربهم وأولئك هم المفلحون", 1),
    #             (
    #                 "../2.mp3",
    #                 "إن الذين كفروا سواء عليهم أأنذرتهم أم لم تنذرهم لا يؤمنون",
    #                 2,
    #             ),
    #             (
    #                 "../3.mp3",
    #                 "ختم الله على قلوبهم وعلى سمعهم وعلى أبصارهم غشاوة ولهم عذاب عظيم",
    #                 3,
    #             ),
    #             (
    #                 "../4.mp3",
    #                 "ومن الناس من يقول آمنا بالله وباليوم الآخر وما هم بمؤمنين",
    #                 4,
    #             ),
    #             (
    #                 "../5.mp3",
    #                 "يخادعون الله والذين آمنوا وما يخدعون إلا أنفسهم وما يشعرون",
    #                 5,
    #             ),
    #         ]
    #     ),
    #     shuffle=False,
    #     map_to=from_path_to_waveform,
    # )

    # validation_dataset = dataset_from_sequence([])
    pass

    # torch.utils.data.DataLoader()


if __name__ == "__main__":
    main()
