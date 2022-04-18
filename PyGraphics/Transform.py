import numpy     as np;
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

     @property
     def front(self)->vec3:return vec3(self.transformM.m02,
                                       self.transformM.m12,
                                       self.transformM.m22);
     
     @property
     def up(self)->vec3:return vec3(self.transformM.m01,
                                    self.transformM.m11,
                                    self.transformM.m21);

     @property
     def right(self)->vec3:return vec3(self.transformM.m00,
                                       self.transformM.m10,
                                       self.transformM.m20);

     #масштаб по Х
     @property
     def sx(self)->float:
        x =  self.transformM.m00;
        y =  self.transformM.m10;
        z =  self.transformM.m20;
        return np.sqrt(x * x + y * y + z * z);
     #масштаб по Y
     @property
     def sy(self)->float:
        x =  self.transformM.m01;
        y =  self.transformM.m11;
        z =  self.transformM.m21;
        return np.sqrt(x * x+ y * y + z * z); 
     #масштаб по Z
     @property
     def sz(self)->float:
        x =  self.transformM.m02;
        y =  self.transformM.m12;
        z =  self.transformM.m22;
        return np.sqrt(x * x+ y * y + z * z); 
     #установить масштаб по Х
     @sx.setter 
     def sx(self,s_x:float):
        if s_x == 0:return;
        scl =  self.sx;
        self.transformM.m00/=scl/s_x;
        self.transformM.m10/=scl/s_x;
        self.transformM.m20/=scl/s_x;
     #установить масштаб по Y
     @sy.setter 
     def sy(self,s_y:float):
        if s_y == 0:return;
        scl =  self.sy;
        self.transformM.m01/=scl/s_y;
        self.transformM.m11/=scl/s_y;
        self.transformM.m21/=scl/s_y;
     #установить масштаб по Z
     @sz.setter 
     def sz(self,s_z:float):
        if s_z == 0:return;
        scl =  self.sz;
        self.transformM.m02/=scl/s_z;
        self.transformM.m12/=scl/s_z;
        self.transformM.m22/=scl/s_z;

     @property
     def scale(self)->vec3:return vec3(self.sx,self.sy,self.sz);
    
     @scale.setter 
     def scale(self,xyz:vec3):
         self.sx = xyz.x;
         self.sy = xyz.y;
         self.sz = xyz.z;
     
     @property
     def x(self)->float:return self.transformM.m03;
    
     @property
     def y(self)->float:return self.transformM.m13;
     
     @property
     def z(self)->float:return self.transformM.m23;

     @x.setter 
     def x(self,x:float):self.transformM.m03=x;
    
     @y.setter 
     def y(self,y:float):self.transformM.m13=y;

     @z.setter 
     def z(self,z:float):self.transformM.m23=z;

     @property
     def origin(self)->vec3:return vec3(self.x, self.y, self.z);
         
     @origin.setter 
     def origin(self,xyz:vec3):self.x = xyz.x; self.y = xyz.y; self.z = xyz.z;
     
     @property
     def angles(self)->vec3:return self.eulerAngles;

     @angles.setter 
     def angles(self, xyz:vec3):
         if(self.eulerAngles.x==xyz.x and self.eulerAngles.y==xyz.y and self.eulerAngles.z==xyz.z):
             return;
         self.eulerAngles.x = xyz.x;
         self.eulerAngles.y = xyz.y;
         self.eulerAngles.z = xyz.z;

         i = MathUtils.rotX(xyz.x);
         i = MathUtils.mul(i, MathUtils.rotY(xyz.y));
         i = MathUtils.mul(i, MathUtils.rotZ(xyz.z));

         scl  = self.scale;
         orig = self.origin;
         self.transformM = i;
         self.scale = scl;
         self.origin = orig;

     @property
     def ax(self)->float:return self.eulerAngles.x;

     @property
     def ay(self)->float:return self.eulerAngles.y;
     
     @property
     def az(self)->float:return self.eulerAngles.z;
    
     @ax.setter 
     def ax(self, x:float):self.angles = vec3(MathUtils.degToRad(x), self.eulerAngles.y, self.eulerAngles.z);
     
     @ay.setter 
     def ay(self, y:float):self.angles = vec3(self.eulerAngles.x, MathUtils.degToRad(y), self.eulerAngles.z);
     
     @az.setter 
     def az(self, z:float):self.angles = vec3(self.eulerAngles.x, self.eulerAngles.y, MathUtils.degToRad(z));

     def rotM(self)->mat4:
        scl = self.scale;
        return mat4(self.transformM.m00/scl.x, self.transformM.m01/scl.y, self.transformM.m02/scl.z, 0,
                    self.transformM.m10/scl.x, self.transformM.m11/scl.y, self.transformM.m12/scl.z, 0,
                    self.transformM.m20/scl.x, self.transformM.m21/scl.y, self.transformM.m22/scl.z, 0,
                    0, 0, 0, 1);

     def lookAt(self, target:vec3, eye:vec3, up:vec3 = vec3(0,1,0)):
         self.transformM  = MathUtils.lookAt(target,eye,up);
         self.eulerAngles = MathUtils.rotMtoEulerAngles(self.rotM());

     # переводит вектор в собственное пространство координат
     def transformVect(self, vec:vec3, w)->vec3:
         if w == 0:
             return vec3(self.transformM.m00 * vec.x + self.transformM.m01 * vec.y + self.transformM.m02 * vec.z,
                         self.transformM.m10 * vec.x + self.transformM.m11 * vec.y + self.transformM.m12 * vec.z,
                         self.transformM.m20 * vec.x + self.transformM.m21 * vec.y + self.transformM.m22 * vec.z);
         
         return vec3(self.transformM.m00 * vec.x + self.transformM.m01 * vec.y + self.transformM.m02 * vec.z + self.transformM.m03,
                     self.transformM.m10 * vec.x + self.transformM.m11 * vec.y + self.transformM.m12 * vec.z + self.transformM.m13,
                     self.transformM.m20 * vec.x + self.transformM.m21 * vec.y + self.transformM.m22 * vec.z + self.transformM.m23);
     # не переводит вектор в собственное пространство координат =)
     def invTransformVect(self, vec:vec3, w)->vec3:
         if w == 0:
             return vec3(self.transformM.m00 * vec.x + self.transformM.m10 * vec.y + self.transformM.m20 * vec.z,
                         self.transformM.m01 * vec.x + self.transformM.m11 * vec.y + self.transformM.m21 * vec.z,
                         self.transformM.m02 * vec.x + self.transformM.m12 * vec.y + self.transformM.m22 * vec.z);
        
         vec_ = vec3(vec.x - self.x, vec.y - self.y, vec.z -self.z);
         return vec3(self.transformM.m00 * vec_.x + self.transformM.m10 * vec_.y + self.transformM.m20 * vec_.z,
                     self.transformM.m01 * vec_.x + self.transformM.m11 * vec_.y + self.transformM.m21 * vec_.z,
                     self.transformM.m02 * vec_.x + self.transformM.m12 * vec_.y + self.transformM.m22 * vec_.z) 


#row major 2D transform
class transform2(object):
     def __init__(self):
         self.transformM  = mat3(1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0);  
         self.zAngle = 0.0;

     @property
     def front(self)->vec2:return vec2(self.transformM.m00,self.transformM.m10);
     
     @property
     def up(self)->vec2:return vec2(self.transformM.m01,self.transformM.m11);

     @property
     def scale(self)->vec2:return vec2(self.sx, self.sy);
     #масштаб по Х
     @property
     def sx(self)->float:
        x =  self.transformM.m00;
        y =  self.transformM.m10;
        return np.sqrt(x * x + y * y);
     #масштаб по Y
     @property
     def sy(self)->float:
        x =  self.transformM.m01;
        y =  self.transformM.m11;
        return np.sqrt(x * x + y * y); 
     #установить масштаб по Х
     @sx.setter 
     def sx(self,s_x:float):
        if s_x == 0:return;
        scl =  self.sx;
        self.transformM.m00/=scl/s_x;
        self.transformM.m10/=scl/s_x;
     #установить масштаб по Y
     @sy.setter 
     def sy(self,s_y:float):
        if s_y == 0:return;
        scl =  self.sy;
        self.transformM.m01/=scl/s_y;
        self.transformM.m11/=scl/s_y;
     
     @scale.setter 
     def scale(self, sxy:vec2): self.sx = sxy.x; self.sy = sxy.y; 

     @property
     def x(self)->float:return self.transformM.m02;
    
     @property
     def y(self)->float:return self.transformM.m12;

     @property
     def origin(self)->vec2:return vec2(self.x, self.y);
     
     @x.setter 
     def x(self,x:float):self.transformM.m02=x;
    
     @y.setter 
     def y(self,y:float):self.transformM.m12=y;
         
     @origin.setter 
     def origin(self,xy:vec2):self.x = xy.x; self.y = xy.y;
   
     @property
     def az(self)->float: return self.zAngle;
     
     @az.setter 
     def az(self, angle:float):
         if(self.zAngle==angle):return;
         self.zAngle = angle;
         cos_a = np.cos(angle);
         sin_a = np.sin(angle);
         rz = mat3(cos_a,-sin_a, 0,
                   sin_a, cos_a, 0,
                   0,     0,     1);
         scl  = self.scale();
         orig = self.origin();
         self.transformM = rz;
         self.scale = scl;
         self.origin = orig;
     
     # переводит вектор в собственное пространство координат
     def transformVect(self, vec:vec2, w)->vec2:
         if w == 0:
             return vec2(self.transformM.m00 * vec.x + self.transformM.m01 * vec.y,
                         self.transformM.m10 * vec.x + self.transformM.m11 * vec.y);
         
         return vec2(self.transformM.m00 * vec.x + self.transformM.m01 * vec.y + self.transformM.m02,
                     self.transformM.m10 * vec.x + self.transformM.m11 * vec.y + self.transformM.m12);
   
     # не переводит вектор в собственное пространство координат =)
     def invTransformVect(self, vec:vec3, w)->vec3:
         if w == 0:
             return vec2(self.transformM.m00 * vec.x + self.transformM.m10 * vec.y,
                         self.transformM.m01 * vec.x + self.transformM.m11 * vec.y);
        
         vec_ = vec2(vec.x - self.x,vec.y - self.y);
         return vec2(self.transformM.m00 * vec.x + self.transformM.m10 * vec.y,
                     self.transformM.m01 * vec.x + self.transformM.m11 * vec.y);