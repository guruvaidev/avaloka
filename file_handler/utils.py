import os


def convert_type(data: str) -> int | float | str:
    if data.isnumeric():
        return int(data)

    try:
        result = float(data)
    except ValueError:
        return data
    else:
        return result


def is_delta(path: str) -> bool:
    return "_delta_log" in path.split(os.path.sep)
