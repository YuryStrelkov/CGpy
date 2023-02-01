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
        self._t0: float = 0.0
        self._c0: float = 0.0
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
               f"\"t0\"        : {self.t0}\n" \
               f"\"c0\"        : {self.c0}\n" \
               f"}}"

    @property
    def t0(self) -> float:
        return self._t0

    @t0.setter
    def t0(self, val: float) -> None:
        if val < 0:
            return
        self._t0 = val

    @property
    def c0(self) -> float:
        return self._c0

    @c0.setter
    def c0(self, val: float) -> None:
        self._c0 = val

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
        return self.curr_val + (self.curr_val + arg) * 0.5 * dt

    def integrate(self, arg: float, t: float) -> float:
        if self._time_val < 0:
            self._time_val = t
            if self._time_val < self._t0:
                return self._c0
            self._last_val = arg
            self._curr_val = arg
            self._time_delta = 0.0
            return self.curr_val + self._c0

        self._time_delta = t - self._time_val
        self._time_val = t
        if self._time_val < self._t0:
            return self._c0
        val = self.__integration_f(arg, self.time_delta)
        self._last_val = self._curr_val
        self._curr_val = val
        return self.curr_val + self._c0

    def reset(self) -> None:
        self._last_val = 0.0
        self._curr_val = 0.0
        self._time_val = 0.0
        self._time_delta = 0.0


class Integrator2d:
    def __init__(self, start_val: Vec2 = None):
        if start_val is None:
            self._last_val: Vec2 = Vec2(0.0, 0.0)
            self._curr_val: Vec2 = Vec2(0.0, 0.0)
        else:
            self._last_val: Vec2 = start_val
            self._curr_val: Vec2 = start_val
        self._t0: float = 0.0
        self._c0: Vec2 = Vec2(0, 0)
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
               f"\"t0\"        : {self.t0}\n" \
               f"\"c0\"        : {self.c0}\n" \
               f"}}"

    @property
    def t0(self) -> float:
        return self._t0

    @t0.setter
    def t0(self, val: float) -> None:
        if val < 0:
            return
        self._t0 = val

    @property
    def c0(self) -> Vec2:
        return self._c0

    @c0.setter
    def c0(self, val: Vec2) -> None:
        self._c0 = val

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
        return self.curr_val + (self.curr_val + arg) * 0.5 * dt

    def integrate(self, arg: Vec2, t: float) -> Vec2:
        if self._time_val < 0:
            self._time_val = t
            if self._time_val < self._t0:
                return self._c0
            self._last_val = arg
            self._curr_val = arg
            self._time_delta = 0.0
            return self.curr_val + self._c0

        self._time_delta = t - self._time_val
        self._time_val = t
        if self._time_val < self._t0:
            return self._c0
        val = self.__integration_f(arg, self.time_delta)
        self._last_val = self._curr_val
        self._curr_val = val
        return self.curr_val + self._c0

    def reset(self) -> None:
        self._last_val = Vec2(0.0)
        self._curr_val = Vec2(0.0)
        self._time_val = 0.0
        self._time_delta = 0.0


class Integrator3d:

    def __init__(self, start_val: Vec3 = None):
        if start_val is None:
            self._last_val: Vec3 = Vec3(0, 0, 0)
            self._curr_val: Vec3 = Vec3(0, 0, 0)
        else:
            self._last_val: Vec3 = start_val
            self._curr_val: Vec3 = start_val
        self._time_val: float = -1.0
        self._time_delta: float = 0.0
        self._t0: float = 0.0
        self._c0: Vec3 = Vec3(0, 0, 0)
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
               f"\"t0\"        : {self.t0}\n" \
               f"\"c0\"        : {self.c0}\n" \
               f"}}"

    @property
    def t0(self) -> float:
        return self._t0

    @t0.setter
    def t0(self, val: float) -> None:
        if val < 0:
            return
        self._t0 = val

    @property
    def c0(self) -> Vec3:
        return self._c0

    @c0.setter
    def c0(self, val: Vec3) -> None:
        self._c0 = val

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
        return self.curr_val + (self.curr_val + arg) * 0.5 * dt

    def integrate(self, arg: Vec3, t: float) -> Vec3:
        if self._time_val < 0:
            self._time_val = t
            if self._time_val < self._t0:
                return self._c0
            self._last_val = arg
            self._curr_val = arg
            self._time_delta = 0.0
            return self.curr_val + self._c0

        self._time_delta = t - self._time_val
        self._time_val = t
        if self._time_val < self._t0:
            return self._c0
        val = self.__integration_f(arg, self.time_delta)
        self._last_val = self._curr_val
        self._curr_val = val
        return self.curr_val + self._c0

    def reset(self) -> None:
        self._last_val = Vec3(0.0)
        self._curr_val = Vec3(0.0)
        self._time_val = 0.0
        self._time_delta = 0.0


"""
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np
    t = np.linspace(0, 1, 1000)
    integrator = Integrator()
    integrator.t0 = 0.5
    integrator.c0 = 1.0
    y = np.array([integrator.integrate(ti, ti) for ti in t])
    plt.plot(t, y)
    plt.show()
"""
