def get_file_ext(f_name: str) -> str:
    id_ = f_name.rfind('.')
    if id_ == -1:
        return f_name
    return f_name[id_+1::]


def get_file_name(f_name: str) -> str:
    ext_ = f_name.rfind('.')
    dir_1 = f_name.rfind('\\')
    dir_2 = f_name.rfind('/')
    if ext_ == -1:
        ext_ = len(f_name)
    dir_ = max(dir_1, dir_2)
    if dir_ < 0:
        return f_name[0:ext_]
    return f_name[dir_ + 1:ext_]


def get_file_dir(_dir_: str) -> str:
    id_ = -1
    separator = ""
    while True:
        id_ = _dir_.rfind('/')
        separator = '/'
        if id_ != -1:
            break
        id_ = _dir_.rfind('\\')
        separator = '\\'
        if id_ != -1:
            break
        return ""
    return _dir_[0:id_] + separator

if __name__ == '__main__':

    example = "aaa\\bbb\\ccc\\ddd\\info.txt"
    print(get_file_dir(example))
    example1 = "aaa/bbb/ccc/ddd/info.txt"
    print(get_file_dir(example1))
    print(get_file_name(example1))