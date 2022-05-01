import random
from typing import List
from pathlib import Path


def split(lst: List, percent: float):
    assert percent > 0 and percent < 1
    n = len(lst)
    random.shuffle(lst)
    split = int(n * percent)
    return lst[:split], lst[split:]


def to_csv(paths: List[Path], labels: List, csv_file: Path, root: Path):
    with csv_file.open("w") as fl:
        fl.writelines(["image_id,real,fake\n"])
        fl.writelines(
            (
                f"{path.relative_to(root).as_posix()},{int(not label)},{label}\n"
                for path, label in zip(paths, labels)
            )
        )


real = list(Path("SOCOFing/Real").glob("*.BMP"))
fake_easy = list(Path("SOCOFing/Altered/Altered-Easy").glob("*.BMP"))
fake_med = list(Path("SOCOFing/Altered/Altered-Medium").glob("*.BMP"))
fake_hard = list(Path("SOCOFing/Altered/Altered-Hard").glob("*.BMP"))

real_train, real_test = split(real, 0.8)

# there is 3 times more fake data than real data
fake = fake_easy + fake_hard + fake_med
fake_train, fake_test = split(fake, 0.8)

to_csv(
    real_train + fake_train,
    [0] * len(real_train) + [1] * len(fake_train),
    Path("SOCOFing/train.csv"),
    root=Path("SOCOFing"),
)

to_csv(
    real_test + fake_test,
    [0] * len(real_test) + [1] * len(fake_test),
    Path("SOCOFing/test.csv"),
    root=Path("SOCOFing"),
)
