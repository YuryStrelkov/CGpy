import numpy as np;
from   PIL import Image;
from   FrameBuffer import RGB;
import MathUtils 
from   MathUtils import vec2, vec3;
from   Transform import transform2;

class texture(object):
    def __init__(self):
        self.transform:transform2 = transform2();
        self.colors:np.uint8 = [];
        self.width_  = -1;
        self.height_ = -1;
        self.bpp_    =  0;

    def setTile(self, tileX:float,tileY:float): self.transform.setScale(tileX, tileY);

    def setOffset(self,x:float,y:float):self.transform.setOrigin(x,y);
    
    def setRotation(self,angle:float):self.transform.rotate(MathUtils.degToRad(angle));
    
    def getTile(self,tileX:float,tileY:float)->vec2:return self.transform.getScale();

    def getOffset(self,x:float,y:float)->vec2:return self.transform.getOrigin();
    
    def getRotation(self,angle:float):self.transform.zAngle;

    def load(self,origin:str):
        if not(len(self.colors) == 0):del(self.colors); self.width_ =-1; self.height_ = -1; self.bpp_ = 0;
        im = Image.open(origin);
        self.width_, self.height_ = im.size;
        self.bpp_ = im.layers;
        self.colors:np.uint8 = (np.asarray(im)).ravel();

    def getColor(self,uv:vec2)->RGB:
        uv_ = self.transform.transformVect(uv, 1);
        uv_x = abs(round(uv_.X * self.width_)  % self.width_);
        uv_y = abs(round(uv_.Y * self.height_) % self.height_);
        pix = (uv_x + uv_y * self.width_ ) * self.bpp_;
        return RGB(self.colors[pix],self.colors[pix + 1],self.colors[pix + 2]);

    @property 
    def width(self)->int:
        return self.width_;
    @property 
    def height(self)->int:
        return self.height_;
    @property 
    def bpp(self)->int:
        return self.bpp_;

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
        return self.diffuse.getColor (uv);
    def normColor(self, uv:vec2)->RGB:
        if self.normals == None: return RGB(255,255,255);
        return self.normals.getColor (uv);
    def specColor(self, uv:vec2)->RGB:
        if self.specular == None: return RGB(255,255,255);
        return self.specular.getColor(uv);




