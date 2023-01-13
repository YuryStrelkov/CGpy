from typing import List


class CircBuffer:
    def _index(self, index: int) -> int:
        return (index + self._indent) % self.capacity

    def __init__(self, cap: int):
        self._indent: int  = 0
        self._n_items: int  = 0
        self._values: List[float] = [0.0 for _ in range(cap)]

    def __getitem__(self, index: int):
        if index >= self.n_items or index < 0:
            raise IndexError(f"CircBuffer :: trying to access index: {index}, while cap is {self.capacity}")
        return self._values[self._index(index)]

    def __setitem__(self, index: int, value):
        if index >= self.n_items or index < 0:
            raise IndexError(f"CircBuffer :: trying to access index: {index}, while cap is {self.capacity}")
        self._values[self._index(index)] = value

    def __str__(self):
        return f"[{', '.join(str(item) for item in self)}]"

    @property
    def n_items(self) -> int:
        return self._n_items

    @property
    def capacity(self) -> int:
        return len(self._values)

    @property
    def sorted(self) -> list:
        return sorted(self._values)

    def append(self, value) -> None:
        self._values[self._index(self.n_items)] = value
        if self.n_items != self.capacity:
            self._n_items += 1
        else:
            self._indent += 1
            self._indent %= self.capacity

    def peek(self) -> float:
        if self.n_items == 0:
            raise IndexError(f"CircBuffer :: pop :: items amount is {self.n_items}")
        value = self._values[self._index(self.n_items - 1)]
        return value

    def pop(self) -> float:
        value = self.peek()
        self._n_items -= 1
        return value

    def clear(self) -> None:
        self._indent = 0
        self._n_items = 0


def buffer_test():
    print(', '.join(str(i % 10) for i in range(-10, 10)))

    buffer = CircBuffer(12)
    buffer.append(1.0)
    buffer.append(2.0)
    buffer.append(3.0)
    print(buffer)
    buffer.append(4.0)
    buffer.append(5.0)
    buffer.append(6.0)
    print(buffer)
    buffer.append(7.0)
    buffer.append(8.0)
    buffer.append(9.0)
    print(buffer)
    buffer.append(10.0)
    buffer.append(11.0)
    buffer.append(12.0)
    print(buffer)

    # print(', '.join(str(buffer[i])for i in range(buffer.n_items)))

    while buffer.n_items != 0:
        print(f"{buffer.pop()}, buffer: {buffer}")

    #buffer.clear()
    #print(buffer)


if __name__ == "__main__":
    # a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # print(a)
    # print(a[-1])
    # print(a[-2])
    buffer_test()

