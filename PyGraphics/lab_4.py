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
       self.uvs        = []
       self.vertices   = []
       self.normals    = []
       self.faces:face = []

    def cleanUp(self):
       if len(self.vertices)==0:return;
       del(self.uvs);
       del(self.vertices);
       del(self.normals);
       del(self.faces);

       self.uvs        = []
       self.vertices   = []
       self.normals    = []
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
        self.translation = np.array([0.005, -0.045, 1.5])
        self.projection = np.array([[5000, 0, 500], [0, 5000, 500], [0, 0, 1]])
        self.rotation: Optional[np.ndarray] = None
        # направление куда смотрит камера
        self.viewForward = np.array([0, 0, 1]);
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
    def draw_trim_polygons(self,color):

        for f in self.obj3D.faces:
            # вершины обрабатываемого треугольника сразу переводим в пространство экрана
            v1 = self.local_space_to_screen(self.obj3D.vertices[f.p_1])
            v2 = self.local_space_to_screen(self.obj3D.vertices[f.p_2])
            v3 = self.local_space_to_screen(self.obj3D.vertices[f.p_3])
            #v1 = self.obj3D.vertices[f.p_1]
            #v2 = self.obj3D.vertices[f.p_2]
            #v3 = self.obj3D.vertices[f.p_3]

            #v1_n = self.obj3D.normals[f.n_1]
            #v2_n = self.obj3D.normals[f.n_2]
            #v3_n = self.obj3D.normals[f.n_3]
            # нормали вершин обрабатываемого треугольника переводим только в мировое пространство
            # тут у вас была ошибка, когда вы считали свет, то вы поворачивали только точки, а не нормали 
            # второй аргумент функции local_space_to_world потому что нормали не сдивгаются 
            v1_n = self.local_space_to_world(self.obj3D.normals[f.n_1], 0)
            v2_n = self.local_space_to_world(self.obj3D.normals[f.n_2], 0)
            v3_n = self.local_space_to_world(self.obj3D.normals[f.n_3], 0)

            l1 = self.viewForward[0] * v1_n[0] + self.viewForward[1] * v1_n[1] + self.viewForward[2] * v1_n[2];
            l2 = self.viewForward[0] * v2_n[0] + self.viewForward[1] * v2_n[1] + self.viewForward[2] * v2_n[2];
            l3 = self.viewForward[0] * v3_n[0] + self.viewForward[1] * v3_n[1] + self.viewForward[2] * v3_n[2];

            # делать отсчечение невидимых граней по направлению падения света неправильно!
            # нужно использовать направление "вперёд" матрицы проекции/камеры (self.viewForward)
            if l1 < 0 and l2 < 0 and l3 < 0:continue;

            #cos = self.trim_polygons(v1, v2, v3)
            #if (cos < 0):
             #   color = (-255 * cos, 0, 0)
            self.draw_triangle_v3(v1, v2, v3, v1_n, v2_n, v3_n, color)
            #self.draw_triangle_solid_color(v1, v2, v3, v1_n, v2_n, v3_n, color);

    # вычисление нормалей
    def calculate_normal(self, x0, y0, z0, x1, y1, z1, x2, y2, z2):
        return np.cross([x1 - x0, y1 - y0, z1 - z0], [x1 - x2, y1 - y2, z1 - z2])

    # отсечение нелицевых граней
    def trim_polygons(self, v1, v2, v3):
        l = np.array([0, 0, 1])
        n = self.calculate_normal(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2])
        return np.dot(n, l) / (np.linalg.norm(n) * np.linalg.norm(l))

    # ЛР 3
    # w = 1.0 - матрица трансфорамции делает поворот и сдвиг(применяется для модификации вершин)
    # w = 0.0 - матрица трансфорамции делает только поворот(применяется для модификации нормалей)
    def local_space_to_world(self, ver, w = 1.0):
        #local space to world space
        if w == 1.0:
            return  np.array([self.rotation[0][0] * ver[0] + self.rotation[0][1] * ver[1] + self.rotation[0][2] * ver[2] + self.translation[0],
                              self.rotation[1][0] * ver[0] + self.rotation[1][1] * ver[1] + self.rotation[1][2] * ver[2] + self.translation[1],
                              self.rotation[2][0] * ver[0] + self.rotation[2][1] * ver[1] + self.rotation[2][2] * ver[2] + self.translation[2]]);
        return  np.array([self.rotation[0][0] * ver[0] + self.rotation[0][1] * ver[1] + self.rotation[0][2] * ver[2],
                          self.rotation[1][0] * ver[0] + self.rotation[1][1] * ver[1] + self.rotation[1][2] * ver[2],
                          self.rotation[2][0] * ver[0] + self.rotation[2][1] * ver[1] + self.rotation[2][2] * ver[2]]);
    
    def local_space_to_screen(self, ver):
        vertex = self.local_space_to_world(ver, 1.0);
        #world space to screen space
        scrCoord = np.array([self.projection[0][0] * vertex[0] + self.projection[0][1] * vertex[1] + self.projection[0][2] * vertex[2],
                             self.projection[1][0] * vertex[0] + self.projection[1][1] * vertex[1] + self.projection[1][2] * vertex[2],
                             self.projection[2][0] * vertex[0] + self.projection[2][1] * vertex[1] + self.projection[2][2] * vertex[2]]);
        # деление на Z компаненту переехало сюды
        scrCoord[0] = scrCoord[0] / scrCoord[2];
        scrCoord[1] = scrCoord[1] / scrCoord[2];
        return scrCoord;
       
    def init_R(self, alpha, beta, gamma):
        pi = 3.141592653589793238462;
        ax = alpha/ 180.0 * pi; 
        ay = beta / 180.0 * pi; 
        az = gamma/ 180.0 * pi; 
        #поворот вокруг ox
        alpha_mat = np.array([ [1,  0,             0],
                               [0,  np.cos(ax), np.sin(ax)],
                               [0, -np.sin(ax), np.cos(ax)]])
        #поворот вокруг oy
        beta_mat = np.array([[np.cos(ay),  0, np.sin(ay)],
                             [0,           1, 0], 
                             [-np.sin(ay), 0, np.cos(ay)]])
        #поворот вокруг oz
        gamma_mat = np.array([[ np.cos(az), np.sin(az),   0],
                              [-np.sin(az), np.cos(gamma),0],
                              [0,              0,         1]])
        self.rotation = np.matmul(np.matmul(alpha_mat, beta_mat), gamma_mat)

    def lights(self, normal):
        l = np.array([0.333, -0.333, 0.333])
        return self.clamp(0,1.0,np.dot(normal, l));# / (np.linalg.norm(normal) * np.linalg.norm(l))

    def clamp(self, min_:float,max_:float,val:float)->float:
        if val<min_:return min_;
        if val>max_:return max_;
        return val;

    def draw_triangle_v3(self, v1, v2, v3, v1n, v2n, v3n, color):
        #v1_p = self.local_space_to_screen(v1)
        #v2_p = self.local_space_to_screen(v2)
        #v3_p = self.local_space_to_screen(v3)
        
        x0 = v1[0];# / v1[2]
        y0 = v1[1];# / v1[2]
        x1 = v2[0];# / v2[2]
        y1 = v2[1];# / v2[2]
        x2 = v3[0];# / v3[2]
        y2 = v3[1];# / v3[2]

        xmin = self.clamp(0,self.width  - 1, min(x0, x1, x2)-1);
        ymin = self.clamp(0,self.height - 1, min(y0, y1, y2)-1);
        xmax = self.clamp(0,self.width  - 1, max(x0, x1, x2)+1);
        ymax = self.clamp(0,self.height - 1, max(y0, y1, y2)+1);

        l1 = (self.lights(v1n))
        l2 = (self.lights(v2n))
        l3 = (self.lights(v3n))
        
        #if l1 < 0 and l2 < 0 and l3 < 0: return;
        bar_div_1 = (x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2);
        bar_div_2 = (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0);
        bar_div_3 = (x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1);

        for i in range(round(xmin) , round(xmax)):
            for j in range(round(ymin) , round(ymax)):
                #bar0, bar1, bar2 = self.baricentr(i, j, x0, x1, x2, y0, y1, y2)
                bar0 = ((x1 - x2) * (j - y2) - (y1 - y2) * (i - x2)) / bar_div_1;
                bar1 = ((x2 - x0) * (j - y0) - (y2 - y0) * (i - x0)) / bar_div_2;
                bar2 = ((x0 - x1) * (j - y1) - (y0 - y1) * (i - x1)) / bar_div_3;
                if bar0 < 0 or bar1 < 0 or bar2 < 0:continue;
                z = bar0 * v1[2] + bar1 * v2[2] + bar2 * v3[2]
                colMult = self.clamp(0.0, 1.0, bar0 * l1 + bar1 * l2 + bar2 * l3);
                if z >= self.zBuffer[i][j]:
                   self.set_pixel(i, j, [int(color[0]*colMult),int(color[1]*colMult), int(color[2]*colMult)])
                   self.zBuffer[i][j] = z

def lab4():
    model = OBJ3DModel()
    model.read("rabbit.obj")
    #model.wrirte(); 
    result = MyImage(model) 
    result.arr_init()
    result.zBuffer_init()
    result.init_R(-15, 45, 0)
    print(result.rotation)
    # лиса
    #result.t = np.array([0.005, -0.045, 1000])
    #result.k = np.array([[5000, 0, 500], [0, 5000, 500], [0, 0, 1]])
    # кролик нахуй
    result.translation = np.array([0.005, -0.045, 1.5])
    result.projection = np.array([[10000, 0, 500], [0, -10000, 500], [0, 0, 1]])
    result.draw_trim_polygons([255, 200, 120])

    result.imshow()

    # result.draw_edges()

