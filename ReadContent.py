# coding:utf-8
from abc import abstractmethod,  ABC
from pathlib import Path


class ReadContent(ABC):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    @abstractmethod
    def update_contents(self):
        pass

# 目录类型1
class ReadContent1(ReadContent):
    def __init__(self, **kwargs):
        super(ReadContent1, self).__init__(**kwargs)

    def update_contents(self):
        fixeds = []
        movings = []
        labels = []
        if self.fixeds_path.is_dir():
            assert self.fixeds_path.is_dir(), "fixeds是目录,而moving非目录"
            dir_queue = [self.movings_path]
            while len(dir_queue) != 0:
                cur_dir = dir_queue.pop()
                for file in cur_dir.glob("*"):
                    if file.is_dir():
                        if file.parent.name == "Image" or file.parent.name == "image":
                            movings.append(file)
                            fixed_path = self.fixeds_path / file.relative_to(self.movings_path)
                            fixeds.append(fixed_path)
                            if self.is_mlabel:
                                filename = file.stem
                                label_path = file.parent.parent / "Mask" / (filename + "_mask.nii")
                                labels.append(label_path)
                        else:
                            dir_queue.insert(0, file)
        else:
            assert self.fixeds_path.is_file(), "fixed是单个文件，而moving是目录"
            assert self.labels_path is None and self.labels_path.is_file(), "fixeds是单个文件，而labesl是目录"
            fixeds.append(self.fixeds_path)
            movings.append(self.movings_path)
            if self.labels_path is not None:
                labels.append(self.labels_path)
        return fixeds, movings, labels

# 目录类型2
class ReadContent2(ReadContent):
    def __init__(self, **kwargs):
        super(ReadContent1, self).__init__(**kwargs)

    def update_contents(self):
        assert self.fixeds_path.is_dir(), "fixeds是目录,而moving非目录"
        fixeds = []
        movings = []
        labels = []
        if self.movings_path.is_dir():
            assert self.fixeds_path.is_dir(), "fixeds是目录,而moving非目录"
            assert self.labels_path is None and self.labels_path.is_dir(), "fixeds是目录,而labels非目录"
            dir_queue = [self.movings_path]
            while len(dir_queue) != 0:
                cur_dir = dir_queue.pop()
                for file in cur_dir.glob("*"):
                    if file.is_dir():
                        dir_queue.insert(0, self.movings_path)
                        continue
                    if self.is_series:  # 序列文件则保存父级目录
                        movings.append(file.parent)
                        fixed_path = self.fixeds_path / file.relative_to(self.movings_path)
                        fixeds.append(fixed_path.parent)
                        if self.is_mlabel:
                            label_path = self.labels_path / file.relative_to(self.movings_path)
                            labels.append(label_path)
                        break
                    else:
                        movings.append(file)
                        fixed_path = self.fixeds_path / file.relative_to(self.movings_path)
                        fixeds.append(fixed_path)
                        if self.is_mlabel:
                            label_path = self.labels_path / file.relative_to(self.movings_path)
                            labels.append(label_path)
        else:
            assert self.fixeds_path.is_file(), "fixed是单个文件，而moving是目录"
            assert self.labels_path is None and self.labels_path.is_file(), "fixeds是单个文件，而labesl是目录"
            fixeds.append(self.fixeds_path)
            movings.append(self.movings_path)
            if self.labels_path is not None:
                labels.append(self.labels_path)

        return fixeds, movings, labels
