import numpy  as     np
from   PIL    import Image
from   typing import Optional, Tuple, Union


def workWithImage(width, height):
    # создание черного изображения
    arr1 = np.zeros((height, width, 3), dtype=np.uint8)
    firstImage = Image.fromarray(arr1, mode="RGB")
    firstImage.save("black.png", mode="RGB")
    # создание белого изображения
    arr2 = np.full((height, width, 3), 255, dtype=np.uint8)
    secondImage = Image.fromarray(arr2, mode="RGB")
    secondImage.save("white.png", mode="RGB")
    # создание цветного изображения
    arr3 = np.full((height, width, 3), (255, 0, 0), dtype=np.uint8)
    thirdImage = Image.fromarray(arr3, mode="RGB")
    thirdImage.save("color.png", mode="RGB")
    # создание градиента
    arr4 = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            arr4[i][j] = ((i + j) % 256, 0, 0)
    fourthImage = Image.fromarray(arr4, mode="RGB")
    fourthImage.save("gradient.png", mode="RGB")


class face:
    def __init__(self):
        self.p_1: int = -1;
        self.uv1: int = -1;
        self.n_1: int = -1;
        self.p_2: int = -1;
        self.uv2: int = -1;
        self.n_2: int = -1;
        self.p_3: int = -1;
        self.uv3: int = -1;
        self.n_3: int = -1;
    def __repr__(self):
        res:str = "<face ";
        res+="%s/%s/%s "%(self.p_1,self.uv1,self.n_1);
        res+="%s/%s/%s "%(self.p_2,self.uv2,self.n_2);
        res+="%s/%s/%s"%(self.p_3,self.uv3,self.n_3);
        res+=">"
        return res;
    def __str__(self):
        res:str = "f [";
        res+="%s/%s/%s "%(self.p_1,self.uv1,self.n_1);
        res+="%s/%s/%s "%(self.p_2,self.uv2,self.n_2);
        res+="%s/%s/%s]"%(self.p_3,self.uv3,self.n_3);
        return res;

class OBJ3DModel(object):
    def __init__(self):
       self.uvs = []
       self.vertices = []
       self.normals = []
       self.faces:face = []

    def cleanUp(self):
       if len(self.vertices)==0:return;
       del(self.uvs);
       del(self.vertices);
       del(self.normals);
       del(self.faces);

       self.uvs = []
       self.vertices = []
       self.normals = []
       self.faces:face = [];

    def wrirte(self):
        if len(self.vertices) == 0:return;
        print("Obj model"); 
        print("vertices number: ",len(self.vertices)); 
        for v in self.vertices:
            print("v  ", v);
        print("normals number: ",len(self.normals)); 
        for n in self.normals:
            print("n ", n);
        print("uvs number: ",len(self.uvs)); 
        for vt in self.uvs:
            print("vt ", vt);
        print("faces number: ",len(self.faces)); 
        for f in self.faces:
            print(f);

    def read(self, path: str):
        self.cleanUp();
        file = open(path)
        tmp:str;
        tmp2:str;
        lines = [];
        for str in file:
            lines.append(str);
        file.close();

        for i in range(len(lines)):
            if len(lines[i])==0:
                continue;
            
            tmp = lines[i].split(" ");
            
            if len(tmp)==0:
                continue;

            if tmp[0] == ("vn"):
                self.normals.append([float(tmp[1]),float(tmp[2]),float(tmp[3])])
            if tmp[0] == ("v"):
                self.vertices.append([float(tmp[1]),float(tmp[2]),float(tmp[3])]);
            if tmp[0] == ("vt"):
                self.uvs.append([float(tmp[1]), float(tmp[2])])
            if tmp[0] == ("f"):
               tmp2 = tmp[1].split("/");
               face_ = face();
               face_.p_1 = int(tmp2[0]) - 1;
               face_.uv1 = int(tmp2[1]) - 1;
               face_.n_1 = int(tmp2[2]) - 1;

               tmp2 = tmp[2].split("/");
               face_.p_2 = int(tmp2[0]) - 1;
               face_.uv2 = int(tmp2[1]) - 1;
               face_.n_2 = int(tmp2[2]) - 1;
                
               tmp2 = tmp[3].split("/");
               face_.p_3 = int(tmp2[0]) - 1;
               face_.uv3 = int(tmp2[1]) - 1;
               face_.n_3 = int(tmp2[2]) - 1;
               self.faces.append(face_);


class MyImage:
    def __init__(self, obj3D: OBJ3DModel):
        self.img_arr: Optional[np.ndarray] = None
        self.width: int = 1000
        self.height: int = 1000
        self.channels: int = 3
        self.delta_t: float = 0.01
        self.obj3D = obj3D;
        self.zBuffer: Optional[np.ndarray] = None
        self.t = np.array([0.005, -0.045, 1.5])
        self.k = np.array([[5000, 0, 500], [0, 5000, 500], [0, 0, 1]])
        self.r: Optional[np.ndarray] = None

    # инициализация массива методом библиотеки numpy
    def arr_init(self):self.img_arr =  np.full((self.height, self.width,self.channels), 200, dtype = np.uint8);#np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

        # инициализация z буфера

    def zBuffer_init(self):self.zBuffer = np.full((self.height, self.width), -np.inf)

    # установка значения цвета пиксела кортежем (tuple) из трех значений R, G, B или одним значением,
    # если изображение одноканальное (полутоновое)
    def set_pixel(self, x: int, y: int, color: Union[Tuple[int, int, int], int] = (255, 0, 0)):
        if x > 0 and y > 0 and x < self.width and y < self.height:
            self.img_arr[y][x] = color

        # конвертация массива в объект класса Image библиотеки Pillow и сохранение его

    # см. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    def save(self, path: str):
        im = Image.fromarray(self.img_arr, mode="RGB")
        im.save(path, mode="RGB")

    # конвертация массива в объект класса Image библиотеки Pillow и вывод на экран
    # см. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.show
    def imshow(self):
        im = Image.fromarray(self.img_arr, mode="RGB")
        im.show()

    # рисование линии, первый вариант алгоритма
    def draw_line_v1(self, x0: int, y0: int, x1: int, y1: int, color: Union[Tuple[int, int, int], int]):
        for t in np.arange(0, 1, self.delta_t):
            x = x0 * (1 - t) + x1 * t
            y = y0 * (1 - t) + y1 * t
            self.set_pixel(int(x), int(y), color)

    # в качестве параметра можно передать саму функцию отрисовки линии
    def draw_star(self, draw_v):
        x0 = 100
        y0 = 100
        for i in range(13):
            alpha = (2 * np.pi * i) / 13
            x1 = 100 + 95 * np.sin(alpha)
            y1 = 100 + 95 * np.cos(alpha)
            draw_v(x0, y0, int(x1), int(y1), (255, 255, 255))

    # рисование линии, второй вариант алгоритма
    def draw_line_v2(self, x0: int, y0: int, x1: int, y1: int, color: Union[Tuple[int, int, int], int]):
        for x in range(x0, x1):
            self.delta_t = (x - x0) / (float(x1 - x0))
            y = y0 * (1 - self.delta_t) + y1 * self.delta_t
            self.set_pixel(int(x), int(y), color)

    # рисование линии, третий вариант алгоритма
    def draw_line_v3(self, x0: int, y0: int, x1: int, y1: int, color: Union[Tuple[int, int, int], int]):
        steep = False
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        for x in range(x0, x1):
            self.delta_t = (x - x0) / (float(x1 - x0))
            y = y0 * (1 - self.delta_t) + y1 * self.delta_t
            if steep:
                self.set_pixel(int(y), int(x), color)
            else:
                self.set_pixel(int(x), int(y), color)

    # рисование линии, четвертый вариант алгоритма (алгоримтм Брезенхема)
    def draw_line_v4(self, x0: int, y0: int, x1: int, y1: int, color: Union[Tuple[int, int, int], int]):
        steep = False
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        dx = x1 - x0
        dy = y1 - y0
        derror = abs(dy / float(dx))
        error = 0.0
        y = y0

        for x in range(int(x0), int(x1)):
            if steep:
                self.set_pixel(int(y), int(x), color)
            else:
                self.set_pixel(int(x), int(y), color)
            error = error + derror
            if error > 0.5:
                y += 1 if y1 > y0 else -1
                error -= 1

    # отрисовка вершин считанной 3D модели
    def draw_vertices(self):
        for point in self.obj3D.vertices:
            x = int(point[0] * 5 + 500)
            y = int(point[1] * 5 + 500)
            self.set_pixel(-x, -y)

    # отрисовка ребер
    def draw_edges(self):
        for f in self.obj3D.faces:
            scale = 5
            shift = 500
            v1 = self.obj3D.vertices[f.p_1]
            v2 = self.obj3D.vertices[f.p_2]
            v3 = self.obj3D.vertices[f.p_2]
            self.draw_line_v4(v1[0] * scale + shift, v1[1] * -scale + shift, v2[0] * scale + shift,
                              v2[1] * -scale + shift, (255, 255, 255))
            self.draw_line_v4(v1[0] * scale + shift, v1[1] * -scale + shift, v3[0] * scale + shift,
                              v3[1] * -scale + shift, (255, 255, 255))
            self.draw_line_v4(v3[0] * scale + shift, v3[1] * -scale + shift, v2[0] * scale + shift,
                              v2[1] * -scale + shift, (255, 255, 255))

    # барицентрические координаты
    def baricentr(self, x, y, x0, x1, x2, y0, y1, y2):
        lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
        lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
        lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        return lambda0, lambda1, lambda2

    # отрисовка полигонов

    def draw_triangle_v1(self):
        for f in self.obj3D.faces:
            scale = 5
            shift = 500
            v1 = self.obj3D.vertices[f.p_1]
            v2 = self.obj3D.vertices[f.p_2]
            v3 = self.obj3D.vertices[f.p_2]
            x0 = v1[0] * scale + shift
            y0 = v1[1] * -scale + shift
            x1 = v2[0] * scale + shift
            y1 = v2[1] * -scale + shift
            x2 = v3[0] * scale + shift
            y2 = v3[1] * -scale + shift

            color = (np.random.choice(range(256)), np.random.choice(range(256)), np.random.choice(range(256)))

            xmin = max(min(x0, x1, x2), 0)
            ymin = max(min(y0, y1, y2), 0)
            xmax = max(x0, x1, x2)
            ymax = max(y0, y1, y2)

            for i in range(round(xmin), round(xmax)):
                for j in range(round(ymin), round(ymax)):
                    bar0, bar1, bar2 = self.baricentr(i, j, x0, x1, x2, y0, y1, y2)
                    if bar0 > 0 and bar1 > 0 and bar2 > 0:
                        self.set_pixel(i, j, color)

    # отрисовка с использованием z-буфера

    def draw_triangle_v2(self, v1, v2, v3, color):
        scale = 5
        shift = 500

        x0 = v1[2] * scale + shift
        y0 = v1[1] * -scale + shift
        x1 = v2[2] * scale + shift
        y1 = v2[1] * -scale + shift
        x2 = v3[2] * scale + shift
        y2 = v3[1] * -scale + shift

        xmin = max(min(x0, x1, x2), 0)
        ymin = max(min(y0, y1, y2), 0)
        xmax = max(x0, x1, x2)
        ymax = max(y0, y1, y2)

        for i in range(round(xmin), round(xmax)):
            for j in range(round(ymin), round(ymax)):
                bar0, bar1, bar2 = self.baricentr(i, j, x0, x1, x2, y0, y1, y2)

                if bar0 > 0 and bar1 > 0 and bar2 > 0:
                    z = bar0 * v1[0] + bar1 * v2[0] + bar2 * v3[0]
                    if z > self.zBuffer[i][j]:
                        self.set_pixel(i, j, color)
                        self.zBuffer[i][j] = z

    # функция отрисовки полигонов с отсечением
    def draw_trim_polygons(self):

        for f in self.obj3D.faces:
            v1 = self.obj3D.vertices[f.p_1]
            v2 = self.obj3D.vertices[f.p_2]
            v3 = self.obj3D.vertices[f.p_3]

            v1_n = self.obj3D.normals[f.n_1]
            v2_n = self.obj3D.normals[f.n_2]
            v3_n = self.obj3D.normals[f.n_3]
            cos = self.trim_polygons(v1, v2, v3)
            if (cos < 0):
                color = (-255 * cos, 0, 0)
                self.draw_triangle_v3(v1, v2, v3, v1_n, v2_n, v3_n, color)

    # вычисление нормалей
    def calculate_normal(self, x0, y0, z0, x1, y1, z1, x2, y2, z2):
        return np.cross([x1 - x0, y1 - y0, z1 - z0], [x1 - x2, y1 - y2, z1 - z2])

    # отсечение нелицевых граней
    def trim_polygons(self, v1, v2, v3):
        l = np.array([0, 0, 1])
        n = self.calculate_normal(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2])
        return np.dot(n, l) / (np.linalg.norm(n) * np.linalg.norm(l))

    # ЛР 3
    def projective(self, ver):
        return np.dot(self.k, np.dot(self.r, ver) + self.t)

    def init_R(self, alpha, beta, gamma):
        alpha_mat = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])
        beta_mat = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        gamma_mat = np.array([[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
        self.r = np.dot(np.dot(alpha_mat, beta_mat), gamma_mat)

    def lights(self, normal):
        l = np.array([1, 0, 0])
        return np.dot(normal, l) / (np.linalg.norm(normal) * np.linalg.norm(l))

    def clamp(self, min_:float,max_:float,val:float)->float:
        if val<min_:return min_;
        if val>max_:return max_;
        return val;

    def draw_triangle_v3(self, v1, v2, v3, v1n, v2n, v3n, color):
        v1_p = self.projective(v1)
        v2_p = self.projective(v2)
        v3_p = self.projective(v3)
        #v1n = self.projective(v1n)
        #v2n = self.projective(v2n)
        #v3n = self.projective(v3n)
        x0 = v1_p[0] / v1_p[2]
        y0 = v1_p[1] / v1_p[2]
        x1 = v2_p[0] / v2_p[2]
        y1 = v2_p[1] / v2_p[2]
        x2 = v3_p[0] / v3_p[2]
        y2 = v3_p[1] / v3_p[2]

        xmin = self.clamp(0,self.width  - 1, min(x0, x1, x2))
        ymin = self.clamp(0,self.height - 1, min(y0, y1, y2))
        xmax = self.clamp(0,self.width  - 1, max(x0, x1, x2))
        ymax = self.clamp(0,self.height - 1, max(y0, y1, y2))

        l1 = (self.lights(v1n))
        l2 = (self.lights(v2n))
        l3 = (self.lights(v3n))
        
        #if l1 < 0 and l2 < 0 and l3 < 0: return;
        bar_div_1 = (x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2);
        bar_div_2 = (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0);
        bar_div_2 = (x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1);

        for i in range(round(xmin), round(xmax)):
            for j in range(round(ymin), round(ymax) ):
                #bar0, bar1, bar2 = self.baricentr(i, j, x0, x1, x2, y0, y1, y2)
                bar0 = ((x1 - x2) * (j - y2) - (y1 - y2) * (i - x2)) / bar_div_1;
                bar1 = ((x2 - x0) * (j - y0) - (y2 - y0) * (i - x0)) / bar_div_2;
                bar2 = ((x0 - x1) * (j - y1) - (y0 - y1) * (i - x1)) / bar_div_2;
                if bar0 > 0 and bar1 > 0 and bar2 > 0:
                    z = bar0 * v1[2] + bar1 * v2[2] + bar2 * v3[2]
                    color = int(255 * (self.clamp(0.0, 1.0, bar0 * l1 + bar1 * l2 + bar2 * l3)))
                    if z > self.zBuffer[i][j]:
                        self.set_pixel(i, j, color)
                        self.zBuffer[i][j] = z


def lab4():
    model = OBJ3DModel()
    model.read("rabbit.obj")
    model.wrirte(); 
    result = MyImage(model)
    result.arr_init()
    result.zBuffer_init()
    result.init_R(0, 0, 0)
    # лиса
    #result.t = np.array([0.005, -0.045, 1000])
    #result.k = np.array([[5000, 0, 500], [0, 5000, 500], [0, 0, 1]])
    result.draw_trim_polygons()

    result.imshow()

    # result.draw_edges()

