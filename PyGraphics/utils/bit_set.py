import ctypes


def set_bit(bytes_: ctypes.c_int8, bit_: int) -> ctypes.c_int8:
    bytes_.value |= (1 << bit_)
    return bytes_


def is_bit_set(bytes_: ctypes.c_int8, bit_: int) -> bool:
    return (bytes_.value & (1 << bit_)) != 0


def inverse_bit(bytes_: ctypes.c_int8, bit_: int) -> ctypes.c_int8:
    bytes_.value ^= (1 << bit_)
    return bytes_


def clear_bit(bytes_: ctypes.c_int8, bit_: int) -> ctypes.c_int8:
    bytes_.value &= ~(1 << bit_)
    return bytes_


class BitSet(ctypes.c_int8):

    __empty_state = ctypes.c_uint8(0)

    __full_state = ctypes.c_uint8(255)

    def __init__(self):
        super().__init__()

    def __str__(self):
        res: str = ""
        for i in range(8):
            if self.is_bit_set(i):
                res += "1"
                continue
            res += "0"
        return res

    @property
    def is_empty(self) -> bool:
        return self == BitSet.__empty_state

    @property
    def is_full(self) -> bool:
        return self == BitSet.__full_state

    def is_bit_set(self, bit_: int):
        return is_bit_set(self, bit_)

    def set_bit(self, bit_: int):
        set_bit(self, bit_)

    def inverse_bit(self, bit_: int):
        inverse_bit(self, bit_)

    def clear_bit(self, bit_: int):
        clear_bit(self, bit_)
