import numpy       as np;
from   PIL         import Image;
import MathUtils 
from   MathUtils   import vec2, vec3;
from   Transform   import transform2;

class RGB(object):
    def __init__(self, r:np.uint8, g:np.uint8, b:np.uint8): self.rgb:np.uint8 = [np.uint8(r), np.uint8(g), np.uint8(b)];
    def __repr__(self):return "<RGB r:%s g:%s b:%s>" % (self.rgb[0], self.rgb[1],self.rgb[2]);
    def __str__(self): return "[%s, %s, %s]" % (self.rgb[0], self.rgb[1],self.rgb[2]);

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

class texture(object):
    def __init__(self, _w:int = 0, _h:int = 0, _bpp:int = 0):
        self.transform:transform2 = transform2();
        self.colors:np.uint8 = [];
        self.width_  = _w;
        self.height_ = _h;
        self.bpp_    = _bpp;
        #self.transform.scale = vec2(_w, -_h);
        self.clearColor();

    @property 
    def width(self)->int: return self.width_;

    @property 
    def height(self)->int: return self.height_;

    @property 
    def bpp(self)->int: return self.bpp_;

    @property
    def texturePixelSize(self):return self.height * self.width;

    @property
    def tile(self,tileX:float, tileY:float)->vec2: return self.transform.scale;

    @property
    def offset(self,x:float, y:float)->vec2: return self.transform.origin;

    @property
    def textureByteSize(self):return self.bpp * self.height * self.width;

    @property
    def rotation(self)->float: return self.transform.az;

    @tile.setter 
    def tile(self, xy:vec2): self.transform.scale = xy;

    @offset.setter 
    def offset(self, xy:vec2):self.transform.origin = xy;

    @rotation.setter 
    def rotation(self, angle:float):self.transform.az = MathUtils.degToRad(angle);

    @property
    def imageData(self)->Image:
        if self.bpp == 3:return Image.frombytes('RGB', (self.width, self.height), self.colors);
        if self.bpp == 4:return Image.frombytes('RGBA',(self.width, self.height), self.colors);

    def load(self, origin:str):
        if not(len(self.colors) == 0):del(self.colors); self.width_ =-1; self.height_ = -1; self.bpp_ = 0;
        im = Image.open(origin);
        self.width_, self.height_ = im.size;
        self.bpp_ = im.layers;
        self.colors:np.uint8 = (np.asarray(im)).ravel();

    def setColor(self, i:int, j:int, color:RGB):
        pix = round((i + j * self.width_ ) * self.bpp_);
        if pix < 0 :return;
        if pix >=  self.width_ * self.height_ * self.bpp_ - 2:return;
        self.colors[pix]     = color.R;
        self.colors[pix + 1] = color.G;
        self.colors[pix + 2] = color.B;

    def getColor(self, i:int, j:int)->RGB:
        pix = round((i + j * self.width_ ) * self.bpp_);
        if pix < 0 :return RGB(0,0,0);
        if pix >=  self.width_ * self.height_ * self.bpp_ - 2:return RGB(0,0,0);
        return RGB(self.colors[pix    ],
                   self.colors[pix + 1],
                   self.colors[pix + 2]);

    #uv:: uv.x in range[0,1], uv.y in range[0,1]
    def setColorUV(self, uv:vec2, color:RGB):
        uv_ = self.transform.invTransformVect(uv, 1);
        pix = round((uv_.x + uv_.y * self.width_ ) * self.bpp_);
        if pix < 0 :return;
        if pix >=  self.width_ * self.height_ * self.bpp_ - 2:return;
        self.colors[pix]     = color.R;
        self.colors[pix + 1] = color.G;
        self.colors[pix + 2] = color.B;

    #uv:: uv.x in range[0,1], uv.y in range[0,1]
    def getColorUV(self, uv:vec2)->RGB:
        uv_ = self.transform.transformVect(uv, 1);
        uv_x = abs(round(uv_.x * self.width_)  % self.width_);
        uv_y = abs(round(uv_.y * self.height_) % self.height_);
        pix = (uv_x + uv_y * self.width_ ) * self.bpp_;
        return RGB(self.colors[pix],self.colors[pix + 1],self.colors[pix + 2]);

    def show(self):
        self.imageData.show();

    def clearColor(self, color: RGB = RGB(125,125,125)):
        if self.textureByteSize == 0:return;
        if len(self.colors) != 0: del(self.colors);
        self.colors = np.zeros((self.height_ * self.width_ * self.bpp), dtype = np.uint8);
        rgb = [color.R,color.G,color.G];
        for i in range(0,len(self.colors)):self.colors[i] = rgb[i % 3];



class material(object):
    def __init__(self):
        self.diffuse :texture = None;
        self.specular:texture = None;
        self.normals :texture = None;

    def setDiff(self,orig:str):
        if self.diffuse == None: self.diffuse = texture();
        self.diffuse.load(orig);

    def setNorm(self,orig:str):
        if self.normals == None: self.normals = texture();
        self.normals.load(orig);
    
    def setSpec(self,orig:str):
        if self.specular == None: self.specular = texture();
        self.specular.load(orig);

    def diffColor(self, uv:vec2)->RGB:
        if self.diffuse == None: return RGB(255,255,255);
        return self.diffuse.getColorUV (uv);

    def normColor(self, uv:vec2)->RGB:
        if self.normals == None: return RGB(255,255,255);
        return self.normals.getColorUV (uv);

    def specColor(self, uv:vec2)->RGB:
        if self.specular == None: return RGB(255,255,255);
        return self.specular.getColorUV(uv);

