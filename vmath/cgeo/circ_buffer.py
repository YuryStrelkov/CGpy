class CircBuffer:
    def __init__(self, cap: int):
        self._capacity: int  = cap
        self._indent: int  = 0
        self._values = [0.0 for _ in range(cap)]

    def __getitem__(self, index: int):
        if index < 0 or index >= self._capacity:
            raise IndexError(f"CircBuffer :: trying to access index: {index}, while cap is {self.capacity}")
        return self._values[(index + self._indent) % self._capacity]

    def __str__(self):
        return f"[{','.join(str(item) for item in self)}]"

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def sorted(self):
        return sorted(self._values)

    def append(self, value):
        self._values[self._indent] = value
        self._indent += 1
        self._indent %= self._capacity

    def clear(self):
        self._indent = 0
        for i in range(self.capacity):
            self._values[i] = 0.0


def buffer_test():
    buffer = CircBuffer(5)
    buffer.append(1.0)
    buffer.append(2.0)
    buffer.append(3.0)
    print(buffer)
    buffer.append(6.0)
    buffer.append(7.0)
    print(buffer)
    buffer.append(8.0)
    buffer.append(9.0)
    print(buffer)
    buffer.append(10.0)
    print(buffer)


if __name__ == "__main__":
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(a)
    print(a[-1])
    print(a[-2])
    buffer_test()

