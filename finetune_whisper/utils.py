from torch.utils.data import (
    IterableDataset,
    Dataset,
    Sampler,
    SequentialSampler,
    RandomSampler,
)
from typing import (
    Iterator,
    Union,
    Optional,
    Callable,
    TypeVar,
    Collection,
    Sequence,
)


T_co = TypeVar("T_co", covariant=True)
U_co = TypeVar("U_co")


class IterableDatasetFromDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: Dataset[U_co],
        sampler: Sampler,
        map_fn: Callable[[U_co], T_co],
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.sampler = sampler
        self.map_fn = map_fn

        if hasattr(dataset, "__getitems__") and callable(
            getattr(dataset, "__getitems__")
        ):
            self.__getitems__ = lambda indices: [
                self.map_fn(elem) for elem in dataset.__getitems__(indices)
            ]

    def __len__(self) -> int:
        if hasattr(self.dataset, "__len__") and callable(
            getattr(self.dataset, "__len__")
        ):
            return len(self.dataset)

        raise ValueError("could not determine the length")

    def __getitem__(self, index) -> T_co:
        return self.map_fn(self.dataset[index])

    def __iter__(self) -> Iterator[T_co]:
        self.sampler_iterator = iter(self.sampler)
        return self

    def __next__(self) -> T_co:
        index = next(self.sampler_iterator)
        return self.map_fn(self.dataset[index])


class IterableDatasetFromIterableDataset(IterableDataset[T_co]):
    def __init__(
        self,
        iterable_dataset: IterableDataset[U_co],
        map_fn: Callable[[U_co], T_co],
    ) -> None:
        super().__init__()
        self.iterable_dataset = iterable_dataset
        self.map_fn = map_fn

    def __len__(self) -> int:
        if hasattr(self.iterable_dataset, "__len__") and callable(
            getattr(self.iterable_dataset, "__len__")
        ):
            return len(self.iterable_dataset)

        raise ValueError("could not determine the length")

    def __iter__(self) -> Iterator[T_co]:
        return map(self.map_fn, self.iterable_dataset)


V_co = TypeVar("V_co", covariant=True)


def prepare_dataset(
    dataset: Union[
        IterableDataset[T_co],  # can't be shuffled
        Dataset[T_co],  # can be shuffled
    ],
    *,
    shuffle: Optional[bool] = None,
    sampler: Optional[Sampler] = None,
    map_to: Optional[Callable[[T_co], V_co]] = None,
) -> Union[IterableDataset[V_co], Dataset[V_co]]:
    r"""
    dataset(dataset, shuffle=None, sampler=None, map_to=None)

    Constructs a dataset that can be used for training or validation

    Args:
        dataset: TODO.

        shuffle(bool, optional): whether shuffling samples is needed or not.

        sampler(torch.utils.data.Sampler, optional): A sampler that will generate the indicies used to get samples of the dataset.

        map_to: TODO.
    """

    # TODO(OmarTairq612): we can't use IterableDataset for validation, I don't know the reason behind this yet!
    # someone may say because we want to validate multiple times, I would respond to him that the training Dataset
    # can be used for many epochs, what would the trainer do in this case if the training dataset was Iterable ??????

    # 1. IterableDataset
    if isinstance(dataset, IterableDataset):
        if shuffle:
            raise ValueError(
                "expected `shuffle` to be unspecified for IterableDatasets"
            )

        if sampler:
            raise ValueError(
                "expected `sampler` to be unspecified for IterableDatasets"
            )

        if map_to:
            return IterableDatasetFromIterableDataset[V_co, T_co](dataset, map_to)

        return dataset  # type: ignore

    # 2.Dataset
    if isinstance(dataset, Dataset):

        def identity(elem):
            return elem

        if sampler:
            if shuffle:
                raise ValueError(
                    "expected `shuffle` to be unspecified when providing a `sampler` for Datasets"
                )

            if map_to:
                return IterableDatasetFromDataset[V_co, T_co](dataset, sampler, map_to)

            return IterableDatasetFromDataset[V_co, T_co](
                dataset, sampler, identity
            )  # assume it is already an IterableDataset[Sample]
        else:
            if not hasattr(dataset, "__len__"):
                raise ValueError("dataset does not have __len__ method")

            if not callable(getattr(dataset, "__len__")):
                raise ValueError("dataset has __len__ but it's not callable")

            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

            if map_to:
                return IterableDatasetFromDataset[V_co, T_co](dataset, sampler, map_to)

            return IterableDatasetFromDataset[V_co, T_co](
                dataset, sampler, identity
            )  # assume it is already an IterableDataset[Sample]

    raise ValueError(
        "dataset must be either a torch.utils.dataset.Dataset or torch.utils.dataset.IterableDataset"
    )


class DatasetFromSequence(Dataset):
    def __init__(self, sequence: Sequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        return self.sequence[index]


class IterableDatasetFromCollection(IterableDataset):
    def __init__(self, collection: Collection):
        self.collection = collection

    def __len__(self):
        return len(self.collection)

    def __iter__(self) -> Iterator:
        return iter(self.collection)


def iterable_dataset_from_collection(collection: Collection) -> IterableDataset:
    return IterableDatasetFromCollection(collection)


def dataset_from_sequence(sequence: Sequence) -> Dataset:
    return DatasetFromSequence(sequence)
