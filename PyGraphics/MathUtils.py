import numpy as np
import math

class vec2(object):
    def __init__(self):self.xy:float = [0, 0];

    def __init__(self, x:float, y:float):self.xy:float = [x, y];
    
    def __repr__(self):return "<vec2 x:%s y:%s>" % (self.xy[0], self.xy[1]);

    def __str__(self):return "[%s, %s]" % (self.xy[0], self.xy[1]);
    
    def __add__(self, other):return vec2(self.X + other.X, self.Y + other.Y);

    def __sub__(self, other):return vec2(self.X - other.X, self.Y - other.Y);

    def __mul__(self, other):return vec2(self.X * other.X,self.Y * other.Y);

    def __truediv__(self, other):return vec2(self.X / other.X,self.Y / other.Y);
    
    def __mul__(self, other:float):return vec2(self.X * other,self.Y * other);

    def __truediv__(self, other:float):return vec2(self.X / other,self.Y / other);

    def norm(self)->float:return np.sqrt(self.xy[0] * self.xy[0] + self.xy[1] * self.xy[1]);

    def normalize(self):
        nrm = self.norm();
        if abs(nrm) < 1e-12:raise ArithmeticError("vec2::zero length vector");
        self.xy[0]/=nrm;
        self.xy[1]/=nrm;

    @property 
    def X(self)->float:return self.xy[0];
    @property 
    def Y(self)->float:return self.xy[1];

    @X.setter 
    def X(self,x:float):self.xy[0] = x;
    @Y.setter 
    def Y(self,y:float):self.xy[1] = y;

def dot(a:vec2, b:vec2)-> float:return a.X * b.X + a.Y * b.Y; 

class vec3(object):
    def __init__(self):self.xyz:float = [0, 0, 0];
    def __init__(self, x:float, y:float, z:float):self.xyz:float = [x, y, z];
    def norm(self)->float:return np.sqrt(self.xyz[0] * self.xyz[0] + self.xyz[1] * self.xyz[1] + self.xyz[2] * self.xyz[2]);
    
    def __repr__(self):return "<vec3 x:%s y:%s z:%s>" % (self.xyz[0], self.xyz[1], self.xyz[2]);
    def __str__(self):return "[%s, %s, %s]" % (self.xyz[0], self.xyz[1], self.xyz[2]);

    def __add__(self, other):return vec3(self.X + other.X, self.Y + other.Y, self.Z + other.Z);
    def __sub__(self, other):return vec3(self.X - other.X, self.Y - other.Y, self.Z - other.Z);

    def __mul__(self, other):return vec3(self.X * other.X,self.Y * other.Y, self.Z / other.Z);
    def __truediv__(self, other):return vec3(self.X / other.X,self.Y / other.Y, self.Z / other.Z);
    
    def __mul__(self, other:float):return vec3(self.X * other, self.Y * other, self.Z * other);
    def __truediv__(self, other:float):return vec3(self.X / other,self.Y / other,self.Z / other);

    def normalize(self):
        nrm = self.norm();
        if abs(nrm) < 1e-12:raise ArithmeticError("zero length vector");
        self.xyz[0]/=nrm;
        self.xyz[1]/=nrm;
        self.xyz[2]/=nrm;

    @property 
    def X(self)->float:return self.xyz[0];
    @property 
    def Y(self)->float:return self.xyz[1];
    @property 
    def Z(self)->float:return self.xyz[2];

    @X.setter 
    def X(self,x:float):self.xyz[0] = x;
    @Y.setter 
    def Y(self,y:float):self.xyz[1] = y;
    @Z.setter 
    def Z(self,z:float):self.xyz[2] = z;

def dot(a:vec3, b:vec3)-> float:return a.X * b.X + a.Y * b.Y +  a.Z *  b.Z; 

def cross(a:vec3, b:vec3)-> vec3:return vec3(a.Z * b.Y - a.Y * b.Z, a.X * b.Z - a.Z * b.X, a.Y * b.X - a.X * b.Y);

class mat3(object):
    def __init__(self):self.data:float = [0,0,0,0,0,0,0,0,0];

    def __init__(self,m0, m1, m2,
                      m3, m4, m5,
                      m6, m7, m8):
        self.data:float = [m0, m1, m2, m3, m4, m5, m6, m7, m8];
    
    def __getitem__(self, key:int)->float:return self.data[key];

    def __setitem__(self, key:int, value:float):self.data[key] = value;
    
    def __repr__(self):
        res:str = "mat4:\n";
        res+="[[%s, %s, %s],\n"%(self.data[0],self.data[1],self.data[2]);
        res+=" [%s, %s, %s],\n"%(self.data[3],self.data[4],self.data[5]);
        res+=" [%s, %s, %s],\n"%(self.data[6],self.data[7],self.data[8]);
        return res;

    def __str__(self):
        res:str = "";
        res+="[[%s, %s, %s],\n"%(self.data[0],self.data[1],self.data[2]);
        res+=" [%s, %s, %s],\n"%(self.data[3],self.data[4],self.data[5]);
        res+=" [%s, %s, %s],\n"%(self.data[6],self.data[7],self.data[8]);
        return res;
    # row 1 set/get
    @property 
    def m00(self)->float:return self.data[0];
    @m00.setter 
    def m00(self,val:float):self.data[0] = val;

    @property 
    def m01(self)->float:return self.data[1];
    @m01.setter 
    def m01(self,val:float):self.data[1] = val;

    @property 
    def m02(self)->float:return self.data[2];
    @m02.setter 
    def m02(self,val:float):self.data[2] = val;
    
    # row 2 set/get
    @property 
    def m10(self)->float:return self.data[3];
    @m10.setter 
    def m10(self,val:float):self.data[3] = val;
   
    @property 
    def m11(self)->float:return self.data[4];
    @m11.setter 
    def m11(self,val:float):self.data[4] = val;

    @property 
    def m12(self)->float:return self.data[5];
    @m12.setter 
    def m12(self,val:float):self.data[5] = val;
    # row 3 set/get
    @property 
    def m20(self)->float:return self.data[6];
    @m20.setter 
    def m20(self,val:float):self.data[6] = val;

    @property 
    def m21(self)->float:return self.data[7];
    @m21.setter 
    def m21(self,val:float):self.data[7] = val;
    
    @property 
    def m22(self)->float:return self.data[8];
    @m22.setter 
    def m22(self,val:float):self.data[8] = val;

def mul(a:mat3, b:mat3) -> mat3:    
    return mat3(
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],

        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],

        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8]
    );

class mat4(object):
    def __init__(self):
        self.data:float = [0,0,0,0,
                           0,0,0,0,
                           0,0,0,0,
                           0,0,0,0];

    def __init__(self,m0, m1, m2, m3,
                      m4, m5, m6, m7,
                      m8, m9, m10,m11,
                      m12,m13,m14,m15):
         self.data:float = [m0, m1, m2, m3,
                            m4, m5, m6, m7,
                            m8, m9, m10,m11,
                            m12,m13,m14,m15];
    
    def __getitem__(self, key:int)->float:return self.data[key];

    def __setitem__(self, key:int, value:float):self.data[key] = value;
    
    def __repr__(self):
        res:str = "mat4:\n";
        res+="[[%s, %s, %s, %s],\n"%(self.data[0],self.data[1],self.data[2],self.data[3]);
        res+=" [%s, %s, %s, %s],\n"%(self.data[4],self.data[5],self.data[6],self.data[7]);
        res+=" [%s, %s, %s, %s],\n"%(self.data[8],self.data[9],self.data[10],self.data[11]);
        res+=" [%s, %s, %s, %s]]"%(self.data[12],self.data[13],self.data[14],self.data[15]);
        return res;

    def __str__(self):
        res:str = "";
        res+="[[%s, %s, %s, %s],\n"%(self.data[0],self.data[1],self.data[2],self.data[3]);
        res+=" [%s, %s, %s, %s],\n"%(self.data[4],self.data[5],self.data[6],self.data[7]);
        res+=" [%s, %s, %s, %s],\n"%(self.data[8],self.data[9],self.data[10],self.data[11]);
        res+=" [%s, %s, %s, %s]]"%(self.data[12],self.data[13],self.data[14],self.data[15]);
        return res;
    # row 1 set/get
    @property 
    def m00(self)->float:return self.data[0];
    @m00.setter 
    def m00(self,val:float):self.data[0] = val;

    @property 
    def m01(self)->float:return self.data[1];
    @m01.setter 
    def m01(self,val:float):self.data[1] = val;

    @property 
    def m02(self)->float:return self.data[2];
    @m02.setter 
    def m02(self,val:float):self.data[2] = val;

    @property 
    def m03(self)->float:return self.data[3];
    @m03.setter 
    def m03(self,val:float):self.data[3] = val;
    # row 2 set/get
    @property 
    def m10(self)->float:return self.data[4];
    @m10.setter 
    def m10(self,val:float):self.data[4] = val;

    @property 
    def m11(self)->float:return self.data[5];
    @m11.setter 
    def m11(self,val:float):self.data[5] = val;

    @property 
    def m12(self)->float:return self.data[6];
    @m12.setter 
    def m12(self,val:float):self.data[6] = val;

    @property 
    def m13(self)->float:return self.data[7];
    @m13.setter 
    def m13(self,val:float):self.data[7] = val;

    # row 3 set/get
    @property 
    def m20(self)->float:return self.data[8];
    @m20.setter 
    def m20(self,val:float):self.data[8] = val;

    @property 
    def m21(self)->float:return self.data[9];
    @m21.setter 
    def m21(self,val:float):self.data[9] = val;

    @property 
    def m22(self)->float:return self.data[10];
    @m22.setter 
    def m22(self,val:float):self.data[10] = val;

    @property 
    def m23(self)->float:return self.data[11];
    @m23.setter 
    def m23(self,val:float):self.data[11] = val;
    # row 4 set/get
    @property 
    def m30(self)->float:return self.data[12];
    @m30.setter 
    def m30(self,val:float):self.data[12] = val;

    @property 
    def m31(self)->float:return self.data[13];
    @m31.setter 
    def m31(self,val:float):self.data[13] = val;

    @property 
    def m32(self)->float:return self.data[14];
    @m32.setter 
    def m32(self,val:float):self.data[14] = val;

    @property 
    def m33(self)->float:return self.data[15];
    @m33.setter 
    def m33(self,val:float):self.data[15] = val;

def mul(a:mat4, b:mat4) -> mat4:
    return mat4(a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12],
                a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13],
                a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14],
                a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15],

                a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12],
                a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13],
                a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14],
                a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15],

                a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12],
                a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13],
                a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14],
                a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15],

                a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12],
                a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13],
                a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14],
                a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15]);

def rotX(angle:float)-> mat4:
    cos_a = np.cos(angle);
    sin_a = np.sin(angle);
    return mat4(1, 0,      0,     0,
                0, cos_a, -sin_a, 0, 
                0, sin_a,  cos_a, 0,
                0, 0,      0,     1);

def rotY(angle:float)-> mat4:
    cos_a = np.cos(angle);
    sin_a = np.sin(angle);
    return mat4(cos_a, 0,  -sin_a, 0,
                0,     1,   0,    0,
                sin_a, 0,   cos_a,    0,
                0,     0,   0,    1);

def rotZ(angle:float)-> mat4:
    cos_a = np.cos(angle);
    sin_a = np.sin(angle);
    return mat4(cos_a,-sin_a, 0, 0,
                sin_a, cos_a, 0, 0,
                0,     0,     1, 0,
                0,     0,     0, 1);

def degToRad(deg:float)->float:return deg / 180.0 * 3.141592653589793238462;

def radToGeg(deg:float)->float:return deg / 3.141592653589793238462 * 180.0;

def rotMtoEulerAngles(rot:mat4)-> vec3:
         if rot.m02 + 1 < 1e-6:return vec3(0,3.141592653589793238462 * 0.5, math.atan2(rot.m10, rot.m20))

         if rot.m02 - 1 < 1e-6:return vec3(0, -3.141592653589793238462 * 0.5, math.atan2(-rot.m10, -rot.m20))

         x1 = -asin(rot.Z);
         x2 = 3.141592653589793238462 - x1;
         y1 = math.atan2(rot.m12 / cos(x1), rot.m22 / cos(x1));
         y2 = math.atan2(rot.m12 / cos(x2), rot.m22 / cos(x2));
         z1 = math.atan2(rot.m01 / cos(x1), rot.m00 / cos(x1));
         z2 = math.atan2(rot.m01 / cos(x2), rot.m00 / cos(x2));
         if (abs(x1) + abs(y1) + abs(z1)) <= (abs(x2) + abs(y2) + abs(z2)):return vec3(x1,y1,z1);
         return vec3(x2,y2,z2);

def lookAt(target:vec3,eye:vec3, up:vec3 = vec3(0,1,0))-> mat4:
         zaxis = target - eye;    # The "forward" vector.
         zaxis.normalize();
         xaxis = cross(up,zaxis);# The "right" vector.
         xaxis.normalize();
         yaxis = cross(zaxis,xaxis);# The "up" vector.

         return mat4(xaxis.X, -yaxis.X, zaxis.X, eye.X,
                    -xaxis.Y, -yaxis.Y, zaxis.Y, eye.Y,
                     xaxis.Z, -yaxis.Z, zaxis.Z, eye.Z,
                     0,       0,     0, 1);

def clamp(min_:float,max_:float,val:float)->float:
    if val<min_:return min_;
    if val>max_:return max_;
    return val;
