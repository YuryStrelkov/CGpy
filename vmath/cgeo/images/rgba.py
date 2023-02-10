from typing import Tuple, List

_r_shift = 0
_g_shift = 8
_b_shift = 16
_a_shift = 24
_mask_r = 255 << _r_shift
_mask_g = _mask_r << _g_shift
_mask_b = _mask_r << _b_shift
_mask_a = _mask_r << _a_shift


def create_color_rgba(red: int, green: int, blue: int, alpha: int) -> int:
    color: int = 0
    color |= ((red & 255) << 0)
    color |= ((green & 255) << 8)
    color |= ((blue & 255) << 16)
    color |= ((alpha & 255) << 24)
    return color


def create_color_rgb(red: int, green: int, blue: int) -> int:
    color: int = 0
    color |= ((red & 255) << 0)
    color |= ((green & 255) << 8)
    color |= ((blue & 255) << 16)
    return color


def _red(value: int) -> int:
    return (value & _mask_r) >> _r_shift


def _green(value: int) -> int:
    return (value & _mask_g) >> _g_shift


def _blue(value: int) -> int:
    return (value & _mask_b) >> _b_shift


def _alpha(value: int) -> int:
    return (value & _mask_a) >> _a_shift


def _set_channel(self: int, value: int, mask: int, shift: int) -> int:
    self &= ~(255 << shift)
    self |= (value & mask) << shift
    return self


def _set_red(value: int, value_red: int) -> int:
    return _set_channel(value, value_red, 255, _r_shift)


def _set_green(value: int, value_green: int) -> int:
    return _set_channel(value, value_green, 255, _g_shift)


def _set_blue(value: int, value_blue: int) -> int:
    return _set_channel(value, value_blue, 255, _b_shift)


def _set_alpha(value: int, value_alpha: int) -> int:
    return _set_channel(value, value_alpha, 255, _a_shift)


class RGBA:
    @staticmethod
    def _unpack_args(*args) -> int:
        args = args[0]
        n_args = len(args)
        if n_args == 0: return 0  # create_color_rgba(125, 127, 130, 0)
        if n_args == 1:  # one argument
            args = args[0]
            if isinstance(args, RGBA):
                return int(args)
            if isinstance(args, int):
                return args  # create_color_rgba(args, args, args, 0)
            if isinstance(args, float):
                c = int(255 * args)
                return create_color_rgba(c, c, c, 0)
        if n_args == 3: return create_color_rgba(args[0], args[1], args[2], 0)
        if n_args == 4: return create_color_rgba(args[0], args[1], args[2], args[3])
        raise TypeError(f'Invalid Input: {args}')

    __slots__ = "_rgba"

    def __init__(self, *args):
        self._rgba: int = RGBA._unpack_args(args)

    def __int__(self):
        return self._rgba

    def __str__(self):
        return f"{{" \
               f"\"red\"  :{self.red:3}, " \
               f"\"green\":{self.green:3}, " \
               f"\"blue\" :{self.blue:3}, " \
               f"\"alpha\":{self.alpha:3}" \
               f"}}"

    def __getitem__(self, index: int):
        if index < 0 or index >= 4:
            raise IndexError(f"RGBA color got only 4 components.\n Trying to access index: {index}")
        return (self._rgba & (255 << (index * 8))) >> (index * 8)

    def __setitem__(self, index: int, value: int):
        if index < 0 or index >= 4:
            raise IndexError(f"RGBA color got only 4 components.\n Trying to access index: {index}")
        self._rgba = _set_channel(self._rgba, max(min(value, 255), 0), 255, 8 * index)

    def __eq__(self, other) -> bool:
        if not (type(other) is RGBA):
            return False
        if not (self._rgba == int(other)):
            return False
        return True

    def __copy__(self):
        color = RGBA()
        color._rgba = self._rgba
        return color

    copy = __copy__

    def __hash__(self) -> int:
        return hash(self._rgba)

    def __iadd__(self, *args):
        color = RGBA._unpack_args(args)
        self._rgba = create_color_rgba(self.red + _red(color),
                                       self.green + _green(color),
                                       self.blue + _blue(color),
                                       self.alpha + _alpha(color))
        return self

    def __add__(self, *args):
        add_color = RGBA._unpack_args(args)
        new_color = self.copy()
        new_color._rgba = create_color_rgba(self.red   + _red(add_color),
                                            self.green + _green(add_color),
                                            self.blue  + _blue(add_color),
                                            self.alpha + _alpha(add_color))
        return new_color

    __radd__ = __add__

    def __isub__(self, *args):
        color = RGBA._unpack_args(args)
        self._rgba = create_color_rgba(max(self.red   - _red(color), 0),
                                       max(self.green - _green(color), 0),
                                       max(self.blue  - _blue(color), 0),
                                       max(self.alpha - _alpha(color), 0))
        return self

    def __sub__(self, *args):
        add_color = RGBA._unpack_args(args)
        new_color = self.copy()
        new_color._rgba = create_color_rgba(max(self.red   - _red(add_color), 0),
                                            max(self.green - _green(add_color), 0),
                                            max(self.blue  - _blue(add_color), 0),
                                            max(self.alpha - _alpha(add_color), 0))
        return new_color

    def __rsub__(self, *args):
        add_color = RGBA._unpack_args(args)
        new_color = self.copy()
        new_color._rgba = create_color_rgba(max(_red(add_color)   - self.red, 0),
                                            max(_green(add_color) - self.green, 0),
                                            max(_blue(add_color)  - self.blue, 0),
                                            max(_alpha(add_color) - self.alpha, 0))
        return new_color

    def __imul__(self, *args):
        color = RGBA._unpack_args(args)
        self._rgba = create_color_rgba(int((self.red   * _red(color)) / 255.0),
                                       int((self.green * _green(color)) / 255.0),
                                       int((self.blue  * _blue(color)) / 255.0),
                                       int((self.alpha * _alpha(color)) / 255.0))
        return self

    def __mul__(self, *args):
        mul_color = RGBA._unpack_args(args)
        new_color = self.copy()
        new_color._rgba = create_color_rgba(int((self.red * _red(mul_color)) / 255.0),
                                            int((self.green * _green(mul_color)) / 255.0),
                                            int((self.blue * _blue(mul_color)) / 255.0),
                                            int((self.alpha * _alpha(mul_color)) / 255.0))
        return new_color

    __rmul__ = __mul__

    @property
    def red(self) -> int:
        return _red(self._rgba)  # (self._rgba & _mask_r) >> _r_shift

    @property
    def green(self) -> int:
        return _green(self._rgba)  # (self._rgba & _mask_g) >> _g_shift

    @property
    def blue(self) -> int:
        return _blue(self._rgba)  # (self._rgba & _mask_b) >> _b_shift

    @property
    def alpha(self) -> int:
        return _alpha(self._rgba)  # (self._rgba & _mask_a) >> _a_shift

    @red.setter
    def red(self, val: int) -> None:
        self._rgba = _set_red(self._rgba, val)

    @green.setter
    def green(self, val: int) -> None:
        self._rgba = _set_green(self._rgba, val)

    @blue.setter
    def blue(self, val: int) -> None:
        self._rgba = _set_blue(self._rgba, val)

    @alpha.setter
    def alpha(self, val: int) -> None:
        self._rgba = _set_alpha(self._rgba, val)

    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.red, self.green, self.blue, self.alpha

    @property
    def as_list(self) -> List[int]:
        return [self.red, self.green, self.blue, self.alpha]

    def to_hex_rgb(self) -> str:
        return ''.join('{:02X}'.format(max(min(a, 255), 0)) for a in (self.red, self.green, self.blue))

    def to_hex(self) -> str:
        return ''.join('{:02X}'.format(max(min(a, 255), 0)) for a in (self.red, self.green, self.blue, self.alpha))

    def to_bin(self) -> str:
        return bin(self._rgba)

"""
if __name__ == "__main__":
    color1 = RGBA()
    color1[0] = 10
    color1[1] = 33
    color1[2] = 12
    color2 = RGBA()
    color2.red = 22
    color2.green = 23
    color2.blue = 13

    print(f"int color1: {int(color1)}\nint color2: {int(color2)}")
    print(f"hex color1: {color1.to_hex()}\nhex color2: {color2.to_hex()}")
    print(f"bin color1: {color1.to_bin()}\nbin color2: {color2.to_bin()}")

    print(f"color1           : {color1}")
    print(f"color2           : {color2}")

    print(f"color2 - color1  : {color2 - color1}")
    print(f"color1 - color2  : {color1 - color2}")

    print(f"color2 + color1  : {color1 + color2}")
    color2 += color1
    print(f"color2 * color1  : {color1 * color2}")
    color1 *= color2
    print(f"color2 *= color1 : {color1}")

    def color_get(n: int) -> Tuple[int, ...]:
        if n == 1:
            return 128,
        if n == 3:
            return 128, 138, 148
        if n == 4:
            return 128, 138, 148, 10
    print(RGBA(*color_get(1)))
    print(RGBA(*color_get(3)))
    print(RGBA(*color_get(4)))


                # print(f"color2 += color1: {color2}")
"""
