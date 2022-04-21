import numpy     as     np
from   PIL       import Image
from   Material  import texture, RGB;
from   MathUtils import vec2;
class frameBuffer(object):
    def __init__(self, w:int, h:int):
        self.colorTexture = texture(w,h,3);
        self.clearColor();
        self.clearDepth();

    @property
    def width(self):return self.colorTexture.width;
    @property
    def height(self):return self.colorTexture.height;

    # инициализация z буфера
    def clearDepth(self):self.zBuffer = np.full((self.height * self.width), -np.inf);

    def clearColor(self, color: RGB = RGB(255, 255, 255)):
        self.colorTexture.clearColor();

    def setPixelUV(self, uv:vec2, color:RGB = RGB(255, 0, 0)):
        self.colorTexture.setColorUV(uv,color);

    def setPixel(self, x:int, y:int, color:RGB = RGB(255, 0, 0)):
        self.colorTexture.setColor(x, y, color);

    def setDepth(self, x: int, y: int, depth:float)->bool:
        pix:int =  self.width * y + x;
        if pix < 0:
            return False;
        if pix >= self.colorTexture.texturePixelSize:
            return False;
        if self.zBuffer[pix] > depth:
            return False;
        self.zBuffer[pix] = depth;
        return True;

    # конвертация массива в объект класса Image библиотеки Pillow и сохранение его
    # см. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    def save(self, path: str):
        im = Image.fromarray(self.img_arr, mode="RGB")
        im.save(path, mode="RGB")
    @property
    def frameBufferImage(self)->Image:
        return self.colorTexture.imageData;
    # конвертация массива в объект класса Image библиотеки Pillow и вывод на экран
    # см. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.show
    def imshow(self):
        self.colorTexture.show();
  
