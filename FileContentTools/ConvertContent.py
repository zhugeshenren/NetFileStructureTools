"""
    example :
        info = dict()
        info["in_path"] = "D:/DataSet/VISION-6-Frame-noise"
        info["out_path"] = {
            "path-1": {"path": "E:/data/train/*?-2{1}*/*?-1*",
                   "proportion": "0.6", "re": [r".npy", r"D[0-9]+"]},
            "path-2": {"path": "E:/data/test/*?-2{1}*/*?-1*",
                   "proportion": "0.2", "re": [r".npy", r"D[0-9]+"]},
            "path-3": {"path": "E:/data/val/*?-2{1}*/*?-1*",
                   "proportion": "0.2", "re": [r".npy", r"D[0-9]+"]}
        }
        info["suffix"] = [".npy", ".png"]
        # random/liner
        info["partitions"] = "random"

        base_context = BaseContext(info)
        out_path = base_context.build()
        print(out_path[0])

    output >> ('D:/DataSet/VISION-6-Frame-noise/train/D19_Apple_iPhone6Plus/D19_V_indoor_panrot_0001-copy_249_I_2.npy',
                'E:/data/train/D19/D19_V_indoor_panrot_0001-copy_249_I_2.npy')

    info = {
        "in_path": "",
        "out_path": {"path": "", "proportion": "", "re": [r"", r""]},
        "suffix": [".npy", ".png"],
        "partitions": "random/liner"
    }
    ** 两个**之间包裹着需要被识别的命令
    ? 表示需要引用输入路径的索引
    : 表示从后面的数字开始自动编号
    {} 表示的是对正则表达式的引用
"""

import copy
import os
import re
import numpy as np


class MaskOperation(object):
    def __init__(self, left, right, opt, re_list):
        self.__count = -1
        self.left = left
        self.right = right
        self.opt = opt

        number_re = re.compile(r"[?:]+-?[0-9]+")
        self.opt_num = re.search(number_re, opt).group()[1:]
        self.opt_num = int(self.opt_num)
        re_re = re.compile(r"{[0-9]+}")
        self.opt_re = re.search(re_re, opt)
        if self.opt_re is not None:
            self.opt_re = self.opt_re.group()[1: -1]
            self.opt_re = re_list[int(self.opt_re)]
            self.opt_re = re.compile(self.opt_re)

        if opt[1] == "?":
            self.opt_type = 0
        elif opt[1] == ":":
            self.opt_type = 1

    def get_string(self, value1):
        self.__count += 1
        if self.opt_type == 0:
            if self.opt_re is not None:
                res = re.search(self.opt_re, value1[self.opt_num])
                if res is None:
                    return ""
                return res.group()
            return value1[self.opt_num]

        elif self.opt_type == 1:

            return str(self.opt_num + self.__count)
        else:
            raise Exception("Error Command")


class PathMask(object):
    def __init__(self, out_path, re_list):

        pattern = r"\*[:\?\-0-9\{\}]*\*"

        self.temp_list = out_path.split("/")
        self.replace_temp = copy.deepcopy(self.temp_list)

        self.operation_list = [list() for _ in range(len(self.temp_list))]
        self.operation_list_2 = [list() for _ in range(len(self.temp_list))]

        pattern = re.compile(pattern)
        for i, path in enumerate(self.temp_list):
            res = re.finditer(pattern, path)
            for r in res:
                self.operation_list[i].append(MaskOperation(r.span()[0], r.span()[1], r.group(), re_list))

    def __replace_temp(self, path_list, index, operations):
        offset = 0
        template = self.temp_list[index]
        for item in operations:
            value = item.get_string(path_list)
            left = item.left
            right = item.right
            left_s = template[0:offset + left]
            right_s = template[offset + right: len(template)]
            offset += len(value) - (right - left)
            template = left_s + value + right_s
        return template

    def fill(self, path_list):

        for index, opt in enumerate(self.operation_list):
            tmp = self.__replace_temp(path_list, index, opt)
            self.replace_temp[index] = tmp

        res = ""
        for item in self.replace_temp:
            res += item + "/"
        return res[:-1]


class BaseContext(object):
    def __init__(self, info):
        # 将所有的信息抽象为一个字典
        self.info = info
        self.partitions = self.info["partitions"]
        self.out_path = self.info["out_path"]
        self.in_file = list()

        self.__search_file()

    def __sub_search_file(self, path):
        if os.path.isfile(path):
            if os.path.splitext(path)[1] not in self.info["suffix"]:
                return
            self.in_file.append(path)
        else:
            paths = os.listdir(path)
            for file_name in paths:
                self.__sub_search_file(path + "/" + file_name)

    def __search_file(self):
        root_path = self.info["in_path"]
        self.__sub_search_file(root_path)
        tmp_list = list()
        for path in self.in_file:
            tmp_list.append(path.split("/"))
        self.in_file = tmp_list

    def __check_out_path(self):
        count = 0
        for key in self.out_path.keys():
            path_info = self.out_path[key]
            proportion = path_info["proportion"]
            count += float(proportion)
        if count > 1:
            raise Exception("overall proportion over 1")

    def __split_path(self):
        pass

    def build(self):
        # 校验
        # 切割list
        # 识别通配符
        # 填充路径
        self.__check_out_path()
        index_list = np.arange(len(self.in_file))
        if self.info["partitions"] == "random":
            np.random.shuffle(index_list)
        elif self.info["partitions"] == "liner":
            pass
        else:
            raise Exception("partitions Error")

        offset = 0.0
        begin = 0
        data_count = len(index_list)
        outs = dict()
        for key in self.out_path.keys():
            out_path = list()
            value = self.out_path[key]
            path_mask = PathMask(value["path"], value["re"])
            proportion = value["proportion"]
            offset += float(proportion)
            end = int(data_count * offset)

            for i in range(begin, end):
                index = index_list[i]
                tmp = path_mask.fill(self.in_file[index])
                in_path = ""
                for p in self.in_file[index]:
                    in_path += p + "/"
                out_path.append((in_path[:-1], tmp))
            outs[key] = out_path
            begin = end

        return outs
