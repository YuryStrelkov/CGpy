from   Transform import transform;
from   MathUtils import vec3, vec2;
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

class meshData(object):
    def __init__(self):
       self.transform_:transform = transform();
       self.uvs:vec2 = []
       self.vertices:vec3 = []
       self.normals:vec3 = []
       self.faces:face = []
       self.max_:vec3 = vec3(-1e12,-1e12,-1e12);
       self.min_:vec3 = vec3( 1e12, 1e12, 1e12);
  
    @property 
    def transformation(self)->transform:return self.transform_;

    @property 
    def maxLocalSpace(self)->vec3:return self.max_;

    @property 
    def minLocalSpace(self)->vec3:return self.min_;

    @property 
    def sizeLocalSpace(self)->vec3:return self.max_ - self.min_;

    @property 
    def centrLocalSpace(self)->vec3:return (self.max_ + self.min_) * 0.5;

    @property 
    def centrWorldSpace(self)->vec3:return self.transformation.transformVect(self.centrLocalSpace,1);

    @property 
    def minWorldSpace(self)->vec3:return self.transformation.transformVect(self.min_,1);
    
    @property 
    def maxWorldSpace(self)->vec3:return self.transformation.transformVect(self.max_,1);

    @property 
    def sizeWorldSpace(self)->vec3:return self.transformation.transformVect(self.sizeLocalSpace,0);

    def cleanUp(self):
       if len(self.vertices)==0:return;
       del(self.uvs);
       del(self.vertices);
       del(self.normals);
       del(self.faces);

       self.uvs:vec2 = []
       self.vertices:vec3 = []
       self.normals:vec3 = []
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
                self.normals.append(vec3(float(tmp[1]),float(tmp[2]),float(tmp[3])))
            if tmp[0] == ("v"):
                v = vec3(float(tmp[1]),float(tmp[2]),float(tmp[3]));
                #update max bound
                if v.x > self.max_.x:self.max_.x = v.x;
                if v.y > self.max_.y:self.max_.y = v.y;
                if v.z > self.max_.z:self.max_.z = v.z;
                #update min bound
                if v.x < self.min_.x:self.min_.x = v.x;
                if v.y < self.min_.y:self.min_.y = v.y;
                if v.z < self.min_.z:self.min_.z = v.z;  

                self.vertices.append(v);
            if tmp[0] == ("vt"):
                self.uvs.append(vec2(float(tmp[1]), float(tmp[2])))
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

    def getVertLocalSpace(self, vert_id):return self.vertices[vert_id];

    def getNormalLocalSpace(self, normal_id):return self.normals[normal_id];

    def getVertWorldSpace(self, vert_id): return self.transform_.transformVect(self.getVertLocalSpace(vert_id), 1);
    
    def getNormalWorldSpace(self, normal_id):
       n:vec3 = self.transform_.transformVect(self.getNormalLocalSpace(normal_id), 0);
       n.normalize();
       return n;


