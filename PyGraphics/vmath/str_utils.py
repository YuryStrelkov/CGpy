from enum import Enum


class StringStartOrigin(Enum):
    LEFT = 0
    RIGHT = 1
    CENTER = 2


def create_empty_str(cap: int = 20, filler: str = " ") -> str:
    res = ""
    for _i in range(0, cap):
        res += filler
    return res


def format_str(value, str_cap: int = 30, origin: StringStartOrigin = StringStartOrigin.LEFT, separator: str = " ") -> str:
    if str_cap % 2 != 0:
        str_cap += 1
    str_val = str(value)
    if origin == StringStartOrigin.LEFT:
        if len(str_val) < str_cap:
            return str_val + create_empty_str(str_cap - len(str_val), separator)
        return str_val

    if origin == StringStartOrigin.RIGHT:
        if len(str_val) < str_cap:
            return create_empty_str(str_cap - len(str_val), separator)+str_val
        return str_val

    if origin == StringStartOrigin.CENTER:
        if len(str_val) >= str_cap:
            return str_val
        free_space: int = str_cap - len(str_val)
        if free_space % 2 == 0:
            return create_empty_str(free_space >> 1, separator) + str_val + create_empty_str(free_space >> 1, separator)
        return create_empty_str((free_space >> 1) + 1, separator) + str_val + create_empty_str(free_space >> 1, separator)


if __name__ == "__main__":
    print(format_str(1.335, 10, StringStartOrigin.LEFT, "_"))
    print(format_str(1.335, 10, StringStartOrigin.RIGHT, "_"))
    print(format_str(1.335, 10, StringStartOrigin.CENTER, "_"))
