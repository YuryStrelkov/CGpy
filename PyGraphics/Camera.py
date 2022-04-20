import numpy       as np;
from   Transform   import transform;
from   MathUtils   import vec3, vec2, mat4;
from   FrameBuffer import frameBuffer;
# определяет направление и положение с которого мы смотрим на 3D сцену
# определяет так же перспективное искажение
class camera(object):
     def __init__(self):
         self.lookAtTransform:transform = transform();
         self.zfar  = 1000;
         self.znear = 0.01;
         self.fov = 60;
         self.aspect = 1;
         self.buildPjojection();

     #Строит матрицу перспективного искажения
     def buildPjojection(self):
         self.projection = mat4(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0,
                                0, 0, 0, 1);
         scale = 1.0 / np.tan(self.fov * 0.5 * 3.1415 / 180); 
         self.projection.m00 = scale * self.aspect; # scale the x coordinates of the projected point 
         self.projection.m11 = scale; # scale the y coordinates of the projected point 
         self.projection.m22 = -self.zfar / (self.zfar - self.znear); # used to remap z to [0,1] 
         self.projection.m32 = -self.zfar * self.znear / (self.zfar - self.znear); # used to remap z [0,1] 
         self.projection.m23 = -1; # set w = -z 
         self.projection.m33 = 0; 

     #ось Z системы координат камеры 
     @property
     def front(self)->vec3:return self.lookAtTransform.front;

     #ось Y системы координат камеры 
     @property
     def up(self)->vec3:return self.lookAtTransform.up;

     #ось Z системы координат камеры 
     @property
     def right(self)->vec3:return self.lookAtTransform.right;

     #Cтроит матрицу вида
     def lookAt(self, target:vec3, eye:vec3, up:vec3 = vec3(0,1,0)):self.lookAtTransform.lookAt(target, eye, up);

     #Переводит точку в пространстве в собственную систему координат камеры 
     def toCameraSpace(self, v:vec3)->vec3:return self.lookAtTransform.invTransformVect(v,1);

     #Переводит точку в пространстве сперва в собственную систему координат камеры,
     #а после в пространство перспективной проекции
     def toClipSpace(self, vect:vec3)->vec3:
             v = self.toCameraSpace(vect);
             out = vec3(v.x * self.projection.m00 + v.y * self.projection.m10 + v.z * self.projection.m20 + self.projection.m30, 
                        v.x * self.projection.m01 + v.y * self.projection.m11 + v.z * self.projection.m21 + self.projection.m31, 
                        v.x * self.projection.m02 + v.y * self.projection.m12 + v.z * self.projection.m22 + self.projection.m32); 
             w =        v.x * self.projection.m03 + v.y * self.projection.m13 + v.z * self.projection.m23 + self.projection.m33; 
             if w != 1: # normalize if w is different than 1 (convert from homogeneous to Cartesian coordinates)
                out.x /= w; out.y /= w; out.z /= w;
             return out;
def renderCamera(fb:frameBuffer, lookAt:vec3, eye:vec3)->camera:
    cam = camera();
    cam.aspect =  float(fb.height) / fb.width;
    cam.lookAt(lookAt, eye);
    cam.buildPjojection();
    return cam;