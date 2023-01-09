from matplotlib import pyplot as plt

from cgeo import Vec3, Vec2


class Integrator:
    SQUARES: int = 0
    TRAPEZOID: int = 1
    SIMPSON: int = 2

    def __init__(self, start_val: float = 0.0):
        self._last_val: float = start_val
        self._curr_val: float = start_val
        self._time_val: float = -1.0
        self._time_delta: float = 0.0
        self.__mode: int = Integrator.TRAPEZOID
        self.__integration_f = self.__trapezoid_int

    def __call__(self, arg: float, t: float) -> float:
        return self.integrate(arg, t)

    def __str__(self):
        return f"{{\n" \
               f"\"prev_val\"  : {self.prev_val},\n" \
               f"\"curr_val\"  : {self.curr_val},\n" \
               f"\"time_val\"  : {self.time_val},\n" \
               f"\"time_delta\": {self.time_delta}\n" \
               f"}}"

    @property
    def time_delta(self) -> float:
        return self._time_delta

    @property
    def time_val(self) -> float:
        return self._time_val

    @property
    def curr_val(self) -> float:
        return self._curr_val

    @property
    def prev_val(self) -> float:
        return self._last_val

    @property
    def mode(self) -> int:
        return self.__mode

    @mode.setter
    def mode(self, arg: int) -> None:
        if arg == Integrator.SQUARES:
            self.__mode = arg
            self.__integration_f = self.__squares_int
            return
        if arg == Integrator.TRAPEZOID:
            self.__mode = arg
            self.__integration_f = self.__trapezoid_int
            return
        if arg == Integrator.SIMPSON:
            self.__mode = arg
            self.__integration_f = self.__simpson_int
            return

    def __squares_int(self, arg: float, dt: float) -> float:
        return self.curr_val + arg * dt

    def __trapezoid_int(self, arg: float, dt: float) -> float:
        return self.curr_val + (self.curr_val + arg) * 0.5 * dt

    def __simpson_int(self, arg: float, dt: float) -> float:
        return (self.curr_val + arg) * 0.5 * dt

    def integrate(self, arg: float, t: float) -> float:
        if self._time_val < 0:
            self._time_val = t
            self._last_val = arg
            self._curr_val = arg
            self._time_delta = 0.0
            return self.curr_val

        self._time_delta = t - self._time_val
        self._time_val = t
        val = self.__integration_f(arg, self.time_delta)
        self._last_val = self._curr_val
        self._curr_val = val
        return self.curr_val

    def reset(self) -> None:
        self._last_val = Vec3(0.0)
        self._curr_val = Vec3(0.0)
        self._time_val = 0.0
        self._time_delta = 0.0


class Integrator2d:
    def __init__(self, start_val: Vec2 = Vec2(0.0, 0.0)):
        self._last_val: Vec2 = start_val
        self._curr_val: Vec2 = start_val
        self._time_val: float = -1.0
        self._time_delta: float = 0.0
        self.__mode: int = Integrator.TRAPEZOID
        self.__integration_f = self.__trapezoid_int

    def __call__(self, arg: Vec2, t: float) -> Vec2:
        return self.integrate(arg, t)

    def __str__(self):
        return f"{{\n" \
               f"\"prev_val\"  : {self.prev_val},\n" \
               f"\"curr_val\"  : {self.curr_val},\n" \
               f"\"time_val\"  : {self.time_val},\n" \
               f"\"time_delta\": {self.time_delta}\n" \
               f"}}"

    @property
    def time_delta(self) -> float:
        return self._time_delta

    @property
    def time_val(self) -> float:
        return self._time_val

    @property
    def curr_val(self) -> Vec2:
        return self._curr_val

    @property
    def prev_val(self) -> Vec2:
        return self._last_val

    @property
    def mode(self) -> int:
        return self.__mode

    @mode.setter
    def mode(self, arg: int) -> None:
        if arg == Integrator.SQUARES:
            self.__mode = arg
            self.__integration_f = self.__squares_int
            return
        if arg == Integrator.TRAPEZOID:
            self.__mode = arg
            self.__integration_f = self.__trapezoid_int
            return
        if arg == Integrator.SIMPSON:
            self.__mode = arg
            self.__integration_f = self.__simpson_int
            return

    def __squares_int(self, arg: Vec2, dt: float) -> Vec2:
        return self.curr_val + arg * dt

    def __trapezoid_int(self, arg: Vec2, dt: float) -> Vec2:
        return self.curr_val + (self.curr_val + arg) * 0.5 * dt

    def __simpson_int(self, arg: Vec2, dt: float) -> Vec2:
        return (self.curr_val + arg) * 0.5 * dt

    def integrate(self, arg: Vec2, t: float) -> Vec2:
        if self._time_val < 0:
            self._time_val = t
            self._last_val = arg
            self._curr_val = arg
            self._time_delta = 0.0
            return self.curr_val

        self._time_delta = t - self._time_val
        self._time_val = t
        val = self.__integration_f(arg, self.time_delta)
        self._last_val = self._curr_val
        self._curr_val = val
        return self.curr_val

    def reset(self) -> None:
        self._last_val = Vec3(0.0)
        self._curr_val = Vec3(0.0)
        self._time_val = 0.0
        self._time_delta = 0.0


class Integrator3d:

    def __init__(self, start_val: Vec3 = Vec3(0.0)):
        self._last_val: Vec3 = start_val
        self._curr_val: Vec3 = start_val
        self._time_val: float = -1.0
        self._time_delta: float = 0.0
        self.__mode: int = Integrator.TRAPEZOID
        self.__integration_f = self.__trapezoid_int

    def __call__(self, arg: Vec3, t: float) -> Vec3:
        return self.integrate(arg, t)

    def __str__(self):
        return f"{{\n" \
               f"\"prev_val\"  : {self.prev_val},\n" \
               f"\"curr_val\"  : {self.curr_val},\n" \
               f"\"time_val\"  : {self.time_val},\n" \
               f"\"time_delta\": {self.time_delta}\n" \
               f"}}"

    @property
    def time_delta(self) -> float:
        return self._time_delta

    @property
    def time_val(self) -> float:
        return self._time_val

    @property
    def curr_val(self) -> Vec3:
        return self._curr_val

    @property
    def prev_val(self) -> Vec3:
        return self._last_val

    @property
    def mode(self) -> int:
        return self.__mode

    @mode.setter
    def mode(self, arg: int) -> None:
        if arg == Integrator.SQUARES:
            self.__mode = arg
            self.__integration_f = self.__squares_int
            return
        if arg == Integrator.TRAPEZOID:
            self.__mode = arg
            self.__integration_f = self.__trapezoid_int
            return
        if arg == Integrator.SIMPSON:
            self.__mode = arg
            self.__integration_f = self.__simpson_int
            return

    def __squares_int(self, arg: Vec3, dt: float) -> Vec3:
        return self.curr_val + arg * dt

    def __trapezoid_int(self, arg: Vec3, dt: float) -> Vec3:
        return self.curr_val + (self.curr_val + arg) * 0.5 * dt

    def __simpson_int(self, arg: Vec3, dt: float) -> Vec3:
        return (self.curr_val + arg) * 0.5 * dt

    def integrate(self, arg: Vec3, t: float) -> Vec3:
        if self._time_val < 0:
            self._time_val = t
            self._last_val = arg
            self._curr_val = arg
            self._time_delta = 0.0
            return self.curr_val

        self._time_delta = t - self._time_val
        self._time_val = t
        val = self.__integration_f(arg, self.time_delta)
        self._last_val = self._curr_val
        self._curr_val = val
        return self.curr_val

    def reset(self) -> None:
        self._last_val = Vec3(0.0)
        self._curr_val = Vec3(0.0)
        self._time_val = 0.0
        self._time_delta = 0.0


def integrator_test(n_points: int = 1024):
    dx = 1.0 / (n_points - 1)
    x = [dx * i for i in range(n_points)]
    print(f"sum(x) = {sum(x) * dx}")
    integrator = Integrator3d()
    integrator.mode = 1
    y = []
    for xi in x:
        y.append(integrator(Vec3(xi), xi).x)

    plt.plot(x, x, 'r')
    plt.plot(x, y, 'g')
    plt.show()