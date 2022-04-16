import numpy as np;
import math;
import MathUtils;
from   MathUtils import vec3, vec2, mat4,mat3;
#row major 3D transform
class transform(object):
     def __init__(self):
         self.transformM  = mat4(1.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0, 0.0,
                                 0.0, 0.0, 0.0, 1.0);  
         self.eulerAngles = vec3(0.0,0.0,0.0);
     # переводит вектор в собственное пространство координат
     def transformVect(self, vec:vec3, w)->vec3:
         if w == 0:
             return vec3(self.transformM.m00 * vec.X + self.transformM.m01 * vec.Y + self.transformM.m02 * vec.Z,
                         self.transformM.m10 * vec.X + self.transformM.m11 * vec.Y + self.transformM.m12 * vec.Z,
                         self.transformM.m20 * vec.X + self.transformM.m21 * vec.Y + self.transformM.m22 * vec.Z);
         
         return vec3(self.transformM.m00 * vec.X + self.transformM.m01 * vec.Y + self.transformM.m02 * vec.Z + self.transformM.m03,
                     self.transformM.m10 * vec.X + self.transformM.m11 * vec.Y + self.transformM.m12 * vec.Z + self.transformM.m13,
                     self.transformM.m20 * vec.X + self.transformM.m21 * vec.Y + self.transformM.m22 * vec.Z + self.transformM.m23);
     # не переводит вектор в собственное пространство координат =)
     def invTransformVect(self, vec:vec3, w)->vec3:
         if w == 0:
             return vec3(self.transformM.m00 * vec.X + self.transformM.m10 * vec.Y + self.transformM.m20 * vec.Z,
                         self.transformM.m01 * vec.X + self.transformM.m11 * vec.Y + self.transformM.m21 * vec.Z,
                         self.transformM.m02 * vec.X + self.transformM.m12 * vec.Y + self.transformM.m22 * vec.Z);
        
         vec_ = vec3(vec.X - self.getX(),vec.Y - self.getY(),vec.Z -self.getZ());
         return vec3(self.transformM.m00 * vec_.X + self.transformM.m10 * vec_.Y + self.transformM.m20 * vec_.Z,
                     self.transformM.m01 * vec_.X + self.transformM.m11 * vec_.Y + self.transformM.m21 * vec_.Z,
                     self.transformM.m02 * vec_.X + self.transformM.m12 * vec_.Y + self.transformM.m22 * vec_.Z) 

     def front(self)->vec3:return vec3(self.transformM.m02,
                                       self.transformM.m12,
                                       self.transformM.m22);
     
     def up(self)->vec3:return vec3(self.transformM.m01,
                                    self.transformM.m11,
                                    self.transformM.m21);

     def right(self)->vec3:return vec3(self.transformM.m00,
                                       self.transformM.m10,
                                       self.transformM.m20);
     #масштаб по Х
     def getSx(self)->float:
        x =  self.transformM.m00;
        y =  self.transformM.m10;
        z =  self.transformM.m20;
        return np.sqrt(x * x+ y * y + z * z);
     #масштаб по Y
     def getSy(self)->float:
        x =  self.transformM.m01;
        y =  self.transformM.m11;
        z =  self.transformM.m21;
        return np.sqrt(x * x+ y * y + z * z); 
     #масштаб по Z
     def getSz(self)->float:
        x =  self.transformM.m02;
        y =  self.transformM.m12;
        z =  self.transformM.m22;
        return np.sqrt(x * x+ y * y + z * z); 
     #установить масштаб по Х
     def setSx(self,s_x:float):
        if s_x == 0:return;
        scl =  self.getSx();
        self.transformM.m00/=scl/s_x;
        self.transformM.m10/=scl/s_x;
        self.transformM.m20/=scl/s_x;
     #установить масштаб по Y
     def setSy(self,s_y:float):
        if s_y == 0:return;
        scl =  self.getSy();
        self.transformM.m01/=scl/s_y;
        self.transformM.m11/=scl/s_y;
        self.transformM.m21/=scl/s_y;
     #установить масштаб по Z
     def setSz(self,s_z:float):
        if s_z == 0:return;
        scl =  self.getSz();
        self.transformM.m02/=scl/s_z;
        self.transformM.m12/=scl/s_z;
        self.transformM.m22/=scl/s_z;

     def setScale(self, sx:float, sy:float, sz:float):
         self.setSx(sx);
         self.setSy(sy);
         self.setSz(sz);

     def getScale(self)->vec3:return vec3(self.getSx(),self.getSy(),self.getSz());
 
     def setX(self,x:float):self.transformM.m03=x;
    
     def setY(self,y:float):self.transformM.m13=y;

     def setZ(self,z:float):self.transformM.m23=z;

     def getX(self)->float:return self.transformM.m03;
    
     def getY(self)->float:return self.transformM.m13;

     def getZ(self)->float:return self.transformM.m23;
         
     def setOrigin(self,x:float,y:float,z:float):
         self.setX(x);
         self.setY(y);
         self.setZ(z);

     def getOrigin(self)->vec3:return vec3(self.getX(), self.getY(), self.getZ());

     def rotate(self, x:float, y:float, z:float):
         if(self.eulerAngles.X==x and self.eulerAngles.Y==y and self.eulerAngles.Z==z):
             return;
         
         self.eulerAngles.X = x;
         self.eulerAngles.Y = y;
         self.eulerAngles.Z = z;

         i = MathUtils.rotX(x);
         i = MathUtils.mul(i, MathUtils.rotY(y));
         i = MathUtils.mul(i, MathUtils.rotZ(z));

         scl  = self.getScale();
         orig = self.getOrigin();
         self.transformM = i;
         self.setScale(scl.X,scl.Y,scl.Z);
         self.setOrigin(orig.X,orig.Y,orig.Z);

     def rotateX(self, x:float):self.rotate(MathUtils.degToRad(x), self.eulerAngles.Y, self.eulerAngles.Z);

     def rotateY(self, y:float):self.rotate(self.eulerAngles.X, MathUtils.degToRad(y), self.eulerAngles.Z);

     def rotateZ(self, z:float):self.rotate(self.eulerAngles.X, self.eulerAngles.Y, MathUtils.degToRad(z));

     def rotM(self)->mat4:
        scl = self.getScale();
        return mat4(self.transformM.m00/scl.X, self.transformM.m01/scl.Y, self.transformM.m02/scl.Z, 0,
                    self.transformM.m10/scl.X, self.transformM.m11/scl.Y, self.transformM.m12/scl.Z, 0,
                    self.transformM.m20/scl.X, self.transformM.m21/scl.Y, self.transformM.m22/scl.Z, 0,
                    0, 0, 0, 1);

     def lookAt(self, target:vec3, eye:vec3, up:vec3 = vec3(0,1,0)):
         self.transformM  = MathUtils.lookAt(target,eye,up);
         self.eulerAngles = MathUtils.rotMtoEulerAngles(self.rotM());
#row major 2D transform
class transform2(object):
     def __init__(self):
         self.transformM  = mat3(1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0);  
         self.zAngle = 0.0;
     # переводит вектор в собственное пространство координат
     def transformVect(self, vec:vec2, w)->vec2:
         if w == 0:
             return vec2(self.transformM.m00 * vec.X + self.transformM.m01 * vec.Y,
                         self.transformM.m10 * vec.X + self.transformM.m11 * vec.Y);
         
         return vec2(self.transformM.m00 * vec.X + self.transformM.m01 * vec.Y + self.transformM.m02,
                     self.transformM.m10 * vec.X + self.transformM.m11 * vec.Y + self.transformM.m12);
     # не переводит вектор в собственное пространство координат =)
     def invTransformVect(self, vec:vec3, w)->vec3:
         if w == 0:
             return vec2(self.transformM.m00 * vec.X + self.transformM.m10 * vec.Y,
                         self.transformM.m01 * vec.X + self.transformM.m11 * vec.Y);
        
         vec_ = vec2(vec.X - self.getX(),vec.Y - self.getY());
         return vec2(self.transformM.m00 * vec.X + self.transformM.m10 * vec.Y,
                     self.transformM.m01 * vec.X + self.transformM.m11 * vec.Y);

     def front(self)->vec2:return vec2(self.transformM.m00,self.transformM.m10);
     
     def up(self)->vec2:return vec2(self.transformM.m01,self.transformM.m11);
     #масштаб по Х
     def getSx(self)->float:
        x =  self.transformM.m00;
        y =  self.transformM.m10;
        return np.sqrt(x * x+ y * y);
     #масштаб по Y
     def getSy(self)->float:
        x =  self.transformM.m01;
        y =  self.transformM.m11;
        return np.sqrt(x * x + y * y); 
     #установить масштаб по Х
     def setSx(self,s_x:float):
        if s_x == 0:return;
        scl =  self.getSx();
        self.transformM.m00/=scl/s_x;
        self.transformM.m10/=scl/s_x;
     #установить масштаб по Y
     def setSy(self,s_y:float):
        if s_y == 0:return;
        scl =  self.getSy();
        self.transformM.m01/=scl/s_y;
        self.transformM.m11/=scl/s_y;

     def setScale(self, sx:float, sy:float):
         self.setSx(sx);
         self.setSy(sy);

     def getScale(self)->vec2:return vec2(self.getSx(),self.getSy());
 
     def setX(self,x:float):self.transformM.m02=x;
    
     def setY(self,y:float):self.transformM.m12=y;

     def getX(self)->float:return self.transformM.m02;
    
     def getY(self)->float:return self.transformM.m12;
         
     def setOrigin(self,x:float,y:float):self.setX(x);self.setY(y);

     def getOrigin(self)->vec2:return vec2(self.getX(), self.getY());

     def rotate(self, angle:float):
         if(self.zAngle==angle):return;
         self.zAngle = angle;
         cos_a = np.cos(angle);
         sin_a = np.sin(angle);
         rz = mat3(cos_a,-sin_a, 0,
                   sin_a, cos_a, 0,
                   0,     0,     1);
         scl  = self.getScale();
         orig = self.getOrigin();
         self.transformM = rz;
         self.setScale(scl.X,scl.Y);
         self.setOrigin(orig.X,orig.Y);