import time


class LoopTimer:
    """
    Интервальный таймер, который можно использоватьв в контенксте with
    """
    __slots__ = "__timeout", "__loop_time", "__time"

    def __init__(self, timeout: float = 1.0):
        if timeout <= 0.001:
            self.__timeout = 0.001
        else:
            self.__timeout = timeout
        self.__loop_time: float = 0.0
        self.__time: float = 0.0

    def __str__(self):
        return f"{{\n" \
               f"\t\"timeout\":        {self.timeout:-1.3f},\n" \
               f"\t\"time\":           {self.time:-1.3f},\n" \
               f"\t\"last_loop_time\": {self.last_loop_time:-1.3f}\n" \
               f"}}"

    def __enter__(self):
        self.__loop_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__loop_time = time.perf_counter() - self.__loop_time
        if self.__loop_time < self.__timeout:
            time.sleep(self.__timeout - self.__loop_time)
            self.__time += self.__timeout
        self.__time += self.__loop_time

    @property
    def time(self) -> float:
        """
        Полное время измеренное таймером
        :return:
        """
        return self.__time

    @property
    def last_loop_time(self) -> float:
        """
        Последнее измеренное время
        :return:
        """
        return self.__loop_time

    @property
    def timeout(self) -> float:
        """
        Время интервала
        :return:
        """
        return self.__timeout

    @timeout.setter
    def timeout(self, val: float) -> None:
        if val < 0.0:
            return
        self.__timeout = val
