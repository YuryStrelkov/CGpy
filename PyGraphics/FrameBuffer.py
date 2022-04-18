import numpy as     np
from   PIL   import Image

class RGB(object):
    def __init__(self):self.rgb:np.uint8 = [255, 255, 255];
    def __init__(self, r:np.uint8, g:np.uint8, b:np.uint8):self.rgb:np.uint8 = [np.uint8(r), np.uint8(g), np.uint8(b)];
    def __repr__(self):return "<RGB r:%s g:%s b:%s>" % (self.rgb[0], self.rgb[1],self.rgb[2]);
    def __str__(self):return "[%s, %s, %s]" % (self.rgb[0], self.rgb[1],self.rgb[2]);

    @property 
    def R(self)->np.uint8:return self.rgb[0];
    @property 
    def G(self)->np.uint8:return self.rgb[1];
    @property 
    def B(self)->np.uint8:return self.rgb[2];

    @R.setter 
    def R(self,r:np.uint8):self.rgb[0] = r;
    @G.setter 
    def G(self,g:np.uint8):self.rgb[1] = g;
    @B.setter 
    def B(self,b:np.uint8):self.rgb[2] = b;

class frameBuffer(object):
    def __init__(self, w:int, h:int):
        self.width:int    = w;
        self.height:int   = h;
        self.channels:int = 3;
        self.pixelsN:int  = self.width * self.height;
        self.clearColor();
        self.clearDepth();

    # инициализация массива методом библиотеки numpy
    def clearColor(self, color: RGB = RGB(125,125,125)):
         self.img_arr = np.full((self.height, self.width, self.channels),(color.R,color.G,color.B), dtype = np.uint8)

    # инициализация z буфера
    def clearDepth(self):self.zBuffer = np.full((self.height, self.width), -np.inf)

    # установка значения цвета пиксела кортежем (tuple) из трех значений R, G, B или одним значением,
    # если изображение одноканальное (полутоновое)
    def setPixel(self, x: int, y: int, color: RGB = RGB(255, 0, 0)):
        if x < 0:return;
        if x >= self.width:return;
        if y < 0:return;
        if y >= self.height:return;
        self.img_arr[y,x,0] = color.R;
        self.img_arr[y,x,1] = color.G;
        self.img_arr[y,x,2] = color.B;
   
    def setDepth(self, x: int, y: int, depth:float):
        if x < 0:return;
        if x >= self.width:return;
        if y < 0:return;
        if y >= self.height:return;
        self.zBuffer[y,x] = depth;
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
  
