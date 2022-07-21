import numpy as np


def set_bit(bytes_: np.uint8, bit_: int) -> np.uint8:
    bytes_ |= (1 << bit_)
    return bytes_


def inverse_bit(bytes_: np.uint8, bit_: int) -> np.uint8:
    bytes_ ^= (1 << bit_)
    return bytes_


def clear_bit(bytes_: np.uint8, bit_: int) -> np.uint8:
    bytes_ &= ~(1 << bit_)
    return bytes_


def is_bit_set(bytes_: np.uint8, bit_: int) -> bool:
    return (bytes_ & (1 << bit_)) != 0


class State:

    __Alive = np.uint8(0)
    __Enable = np.uint8(1)
    __Shown = np.uint8(2)
    __Selected = np.uint8(3)
    __Created = np.uint8(4)

    def __init__(self):
        self.__state: np.uint8 = np.uint8(0)
        self._crete()
        self._birth()

    def __str__(self):
        res: str = ""
        if self.is_alive:
            res += f"[{'alive':10}|"
        else:
            res += f"[{'dead':10}|"

        if self.enable:
            res += f"{'enable':10}|"
        else:
            res += f"{'disable':10}|"

        if self.shown:
            res += f"{'shown':10}|"
        else:
            res += f"{'hide':10}|"

        if self.selected:
            res += f"{'selected':10}]"
        else:
            res += f"{'unselected':10}]"
        return res

    def _crete(self):
        self.__state = set_bit(self.__state, State.__Created)

    def _birth(self):
        self.__state = set_bit(self.__state, State.__Alive)

    def _kill(self):
        self.__state = clear_bit(self.__state, State.__Alive)

    @property
    def enable(self) -> bool:
        return is_bit_set(self.__state, State.__Enable)

    @enable.setter
    def enable(self, value: bool) -> None:
        if value:
            self.__state = set_bit(self.__state, State.__Enable)
            return
        self.__state = clear_bit(self.__state, State.__Enable)

    @property
    def shown(self) -> bool:
        return is_bit_set(self.__state, State.__Shown)

    @shown.setter
    def shown(self, value: bool) -> None:
        if value:
            self.__state = set_bit(self.__state, State.__Shown)
            return
        self.__state = clear_bit(self.__state, State.__Shown)

    @property
    def selected(self) -> bool:
        return is_bit_set(self.__state, State.__Selected)

    @selected.setter
    def selected(self, value: bool) -> None:
        if value:
            self.__state = set_bit(self.__state, State.__Selected)
            return
        self.__state = clear_bit(self.__state, State.__Selected)

    @property
    def is_created(self) -> bool:
        return is_bit_set(self.__state, State.__Created)

    @property
    def is_alive(self) -> bool:
        return is_bit_set(self.__state, State.__Alive)


if __name__ == '__main__':

    s = State()

    print(s)