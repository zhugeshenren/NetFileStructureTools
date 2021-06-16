from FileContentTools.ConvertContent import BaseContext


def main():
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
    info["suffix"] = [".npy"]
    # random/liner
    info["partitions"] = "random"

    base_context = BaseContext(info)
    out_path = base_context.build()
    print(out_path["path-1"])


if __name__ == "__main__":
    main()
