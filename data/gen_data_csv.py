import random
from typing import List
from pathlib import Path

def split(lst: List, percent: float):
    assert percent > 0 and percent < 1
    n = len(lst)
    random.shuffle(lst)
    split = int(n*percent)
    return lst[:split], lst[split:]

def to_csv(paths: List[Path], labels: List, csv_file: Path):
    with csv_file.open('w') as fl:
        fl.writelines(['image_id,healthy,has_tuberculosis\n'])
        fl.writelines(
            (f"{path.as_posix()},{int(not label)},{label}\n" for path, label in zip(paths, labels))
        )

normal = list(Path('tuberculosis/Normal').glob('*.png'))
tb = list(Path('tuberculosis/Tuberculosis').glob('*.png'))

normal_train, normal_test = split(normal, 0.8)
tb_train, tb_test = split(tb, 0.8)

to_csv(
    normal_train+tb_train,
    [0]*len(normal_train) + [1]*len(tb_train),
    Path('train.csv')
)

to_csv(
    normal_test+tb_test,
    [0]*len(normal_test) + [1]*len(tb_test),
    Path('test.csv')
)