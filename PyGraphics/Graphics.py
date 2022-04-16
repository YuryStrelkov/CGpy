import numpy as np
import MathUtils;
from   MathUtils   import vec2,vec3,mat4
from   Material    import material;
from   Camera      import camera
from   FrameBuffer import frameBuffer
from   FrameBuffer import RGB
from   MeshData    import meshData;
class vertex(object):
    def __init__(self,v_:vec3,n_:vec3,uv_:vec2):
        self.v: vec3 = v_;
        self.n: vec3 = n_;
        self.uv:vec2 = uv_;
    def __add__(self, other):return vertex(self.v + other.v, self.n + other.n, self.uv + other.uv);

    def __sub__(self, other):return vertex(self.v - other.v, self.n - other.n, self.uv - other.uv);
    
    def __mul__(self, other):return vertex(self.v * other.v, self.n * other.n, self.uv * other.uv);

    def __truediv__(self, other):return vertex(self.v / other.v, self.n / other.n, self.uv / other.uv);
    
    def __mul__(self, other:float):return vertex(self.v * other, self.n * other, self.uv * other);

    def __truediv__(self, other:float):return vertex(self.v / other, self.n / other, self.uv / other);

def xrange(start, stop=None, step=1):
    if stop is None: stop = start; start = 0;
    else: stop = int(stop)
    start = int(start)
    step = int(step)
    while start < stop:
        yield start
        start += step

def lerpVertex(a:vertex ,b:vertex ,val:float)->vertex:return a + (b - a) * val;
# барицентрические координаты
def baricentric(self, x, y, x0, x1, x2, y0, y1, y2):
    return np.array([((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2)),
                     ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)),
                     ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))])

# рисование линии, первый вариант алгоритма
def drawLineV1(buffer:frameBuffer, x0: int, y0: int, x1: int, y1: int, color:RGB = RGB(255, 255, 255), dt:float = 0.01):
        for t in np.arange(0, 1, dt):
            x = x0 * (1 - t) + x1 * t
            y = y0 * (1 - t) + y1 * t
            buffer.setPixel(int(x), int(y), color)

# рисование линии, второй вариант алгоритма
def drawLineV2(buffer:frameBuffer, x0: int, y0: int, x1: int, y1: int, color: RGB = RGB(255, 255, 255)):
        for x in range(x0, x1):
            dt = (x - x0) / (float(x1 - x0))
            y = y0 * (1 - dt) + y1 * dt
            buffer.setPixel(int(x), int(y), color)

# рисование линии, третий вариант алгоритма
def drawLineV3(buffer:frameBuffer, x0: int, y0: int, x1: int, y1: int, color: RGB = RGB(255, 255, 255)):
        steep = False
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        for x in range(x0, x1):
            dt = (x - x0) / (float(x1 - x0))
            y = y0 * (1 - dt) + y1 * dt
            if steep:
                buffer.setPixel(int(y), int(x), color)
            else:
                buffer.setPixel(int(x), int(y), color)

# рисование линии, четвертый вариант алгоритма (алгоримтм Брезенхема)
def drawLineV4(buffer:frameBuffer, x0: int, y0: int, x1: int, y1: int, color: RGB = RGB(255, 255, 255)):
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
        if dx == 0:return;
        derror = abs(dy / float(dx))
        error = 0.0
        y = y0
        for x in range(int(x0), int(x1)):
            if steep:
                buffer.setPixel(int(y), int(x), color)
            else:
                buffer.setPixel(int(x), int(y), color)
            error = error + derror
            if error > 0.5:
                y += 1 if y1 > y0 else -1
                error -= 1

def drawPoint(buffer:frameBuffer, x:int, y:int, color:RGB = RGB(255, 255, 255), depth:float = 0):
       buffer.setPixel(x - 1, y-1,   color);
       buffer.setPixel(x - 1, y,     color);
       buffer.setPixel(x - 1, y + 1, color);
       buffer.setPixel(x,     y - 1, color);
       buffer.setPixel(x,     y,     color);
       buffer.setPixel(x,     y + 1, color);
       buffer.setPixel(x + 1, y - 1, color);
       buffer.setPixel(x + 1, y,     color);
       buffer.setPixel(x + 1, y + 1, color);
       
       buffer.setDepth(x - 1, y-1,   depth);
       buffer.setDepth(x - 1, y,     depth);
       buffer.setDepth(x - 1, y + 1, depth);
       buffer.setDepth(x,     y - 1, depth);
       buffer.setDepth(x,     y,     depth);
       buffer.setDepth(x,     y + 1, depth);
       buffer.setDepth(x + 1, y - 1, depth);
       buffer.setDepth(x + 1, y,     depth);
       buffer.setDepth(x + 1, y + 1, depth);

def pointToScrSpace(buffer:frameBuffer,pt:vec3)->vec3:
    return vec3( round(MathUtils.clamp(0, buffer.width -1, round(buffer.width  * ( pt.X * 0.5 + 0.5)))),
                 round(MathUtils.clamp(0, buffer.height-1, round(buffer.height * (-pt.Y * 0.5 + 0.5)))),
                pt.Z);

# отрисовка вершин
def drawVertices(buffer:frameBuffer, mesh:meshData, cam:camera = None, color:RGB = RGB(0, 0, 255)):
        if cam == None:cam = camera();cam.lookAt(mesh.minWorldSpace, mesh.maxWorldSpace * 1.5);
        for point in mesh.vertices:
            v1 = pointToScrSpace(buffer, cam.toClipSpace(mesh.transformation.transformVect(point,1)));
            if buffer.zBuffer[v1.X,v1.Y] > v1.Z:return;
            drawPoint(buffer, v1.X,v1.Y, color, v1.Z);

# отрисовка ребер
def drawEdges(buffer:frameBuffer, mesh:meshData, cam:camera = None, color:RGB = RGB(0, 0, 0)):
        if cam == None:cam = camera();cam.lookAt(mesh.minWorldSpace, mesh.maxWorldSpace * 1.5);
        # направление освещения совпадает с направлением взгляда камеры
        forward = cam.front();
        for f in mesh.faces:
            v1 =pointToScrSpace(buffer, cam.toClipSpace(mesh.getVertWorldSpace(f.p_1)));
            v2 =pointToScrSpace(buffer, cam.toClipSpace(mesh.getVertWorldSpace(f.p_2)));
            v3 =pointToScrSpace(buffer, cam.toClipSpace(mesh.getVertWorldSpace(f.p_3)));
            
            n1 = (mesh.getNormalWorldSpace(f.n_1));
            n2 = (mesh.getNormalWorldSpace(f.n_2));
            n3 = (mesh.getNormalWorldSpace(f.n_3));

            a =  -MathUtils.dot(n1, forward);
            b =  -MathUtils.dot(n2, forward);
            c =  -MathUtils.dot(n3, forward);

            if a > 0 or b > 0:drawLineV4(buffer, v1.X, v1.Y, v2.X, v2.Y, color);
            if a > 0 or c > 0:drawLineV4(buffer, v1.X, v1.Y, v3.X, v3.Y, color);
            if b > 0 or c > 0:drawLineV4(buffer, v2.X, v2.Y, v3.X, v3.Y, color);

#отрисовка одноцветного треугольника(интерполируется только глубина) 
def drawTriangleSolid(buffer:frameBuffer, p0:vec3, p1:vec3, p2:vec3, color:RGB = RGB(0, 255, 0)):
    if p0.Y == p1.Y and p0.Y == p2.Y:return; #i dont care about degenerate triangles
    # sort the vertices, p0, p1, p2 lower-to-upper (bubblesort yay!)
    if (p0.Y > p1.Y): p0, p1 = p1, p0;
    if (p0.Y > p2.Y): p0, p2 = p2, p0;
    if (p1.Y > p2.Y): p1, p2 = p2, p1;
    
    total_height:int = round(p2.Y - p0.Y);
    
    for i in xrange( 0, total_height):
         second_half:bool = i > p1.Y-p0.Y or p1.Y == p0.Y;
         
         segment_height:int = p1.Y-p0.Y;
         
         if second_half:segment_height:int = p2.Y-p1.Y;
        
         if segment_height==0:continue;

         alpha:float  = float(i)/total_height;

         beta:float   = 0;

         if second_half: beta = float(i - (p1.Y - p0.Y))/segment_height;
         else:beta = float(i / segment_height)# be careful: with above conditions no division by zero her

         A = p0 + (p2 - p0) * alpha;
         
         if second_half: B = p1 + (p2 - p1)*beta;
         else:B =  p0 + (p1 - p0)*beta;
         
         if (A.X > B.X): A, B = B, A;
         for j in xrange(round(A.X),round(B.X)):
             phi:float = 0.0;
             if B.X==A.X: phi = 1.0
             else: phi = float(j-A.X)/float(B.X-A.X);
             P:vec3  = A + (B - A) * phi;
             zx,xy = round(P.X), round(P.Y);
             if buffer.zBuffer[zx, xy] < P.Z:
                buffer.zBuffer[zx, xy] = P.Z;
                buffer.setPixel(zx, xy, color);

#отрисовка треугольника(интерполируется только глубина, нормали, барицентрические координаты) 
def drawTriangleShaded(buffer:frameBuffer, p0:vertex,  p1:vertex,  p2:vertex, mat:material): #позиции(в прострастве экрана) вершин треугольника
    if p0.v.Y == p1.v.Y and p0.v.Y == p2.v.Y:return; #i dont care about degenerate triangles
    # sort the vertices, p0, p1, p2 lower-to-upper (bubblesort yay!)
    if (p0.v.Y > p1.v.Y): p0, p1   = p1, p0;

    if (p0.v.Y > p2.v.Y): p0, p2   = p2, p0;

    if (p1.v.Y > p2.v.Y): p1, p2   = p2, p1;
    
    total_height:int = round(p2.v.Y - p0.v.Y);
    
    for i in xrange( 0, total_height):
         second_half:bool = i > p1.v.Y-p0.v.Y or p1.v.Y == p0.v.Y;
         
         segment_height:int = p1.v.Y-p0.v.Y;
         
         if second_half:segment_height:int = p2.v.Y-p1.v.Y;
        
         if segment_height==0:continue;

         alpha:float  = float(i)/total_height;

         beta:float   = 0;

         if second_half: beta = float(i - (p1.v.Y - p0.v.Y))/segment_height;
         else:beta = float(i / segment_height)# be careful: with above conditions no division by zero her
         A   = lerpVertex(p0, p2, alpha);
         if second_half: B = lerpVertex(p1, p2, beta);
         else: B = lerpVertex(p0, p1,beta);
         if (A.v.X > B.v.X): A, B = B, A;

         for j in xrange(round(A.v.X),round(B.v.X)):
             phi:float = 0.0;
             if B.v.X==A.v.X: phi = 1.0
             else: phi = float(j-A.v.X)/float(B.v.X-A.v.X);
             P   = lerpVertex(A,B,phi);
             zx,xy = round(P.v.X), round(P.v.Y);
             if buffer.zBuffer[zx, xy] < P.v.Z:
                buffer.zBuffer[zx, xy] = P.v.Z;
                col:RGB = mat.diffColor(P.uv);
                colShading:float = MathUtils.clamp(0.0, 1.0, MathUtils.dot(P.n, vec3(0.333,0.333,0.333)));
                buffer.setPixel(zx, xy, RGB(col.R * colShading, col.G * colShading, col.B * colShading));

# рисует полигональную сетку интерполируя только по глубине и заливает одним цветом
def drawMeshSolidColor(buffer:frameBuffer, mesh:meshData, cam:camera = None, color:RGB = RGB(255, 255, 255)):
    # направление освещения совпадает с направлением взгляда камеры
    if cam == None: cam = camera(); cam.lookAt(mesh.minWorldSpace, mesh.maxWorldSpace * 1.5);
    forward = cam.front();
    for f in mesh.faces:
        # переводим нормали вершин в мировое пространство
        n1 = (mesh.getNormalWorldSpace(f.n_1));
        n2 = (mesh.getNormalWorldSpace(f.n_2));
        n3 = (mesh.getNormalWorldSpace(f.n_3));

        a =  -MathUtils.dot(n1, forward);
        b =  -MathUtils.dot(n2, forward);
        c =  -MathUtils.dot(n3, forward);
        col = (MathUtils.clamp(0.0, 1.0, a) + MathUtils.clamp(0.0, 1.0, b) + MathUtils.clamp(0.0, 1.0, c)) * 0.333333333;
        # если скалярное произведение для всех нормалей на направление взгляда < 0, то не рисуем, тк
        # треугольник к нам задом(back-face culling)
        if col < 0: return;
      # перевоим точки в простраснтво отсечений камеры
        v1 = cam.toClipSpace(mesh.getVertWorldSpace(f.p_1));
        v2 = cam.toClipSpace(mesh.getVertWorldSpace(f.p_2));
        v3 = cam.toClipSpace(mesh.getVertWorldSpace(f.p_3));
        drawTriangleSolid(buffer,pointToScrSpace(buffer, v1),
                                 pointToScrSpace(buffer, v2),
                                 pointToScrSpace(buffer, v3),
                                 RGB(color.R * col, color.G * col, color.B * col));

# рисует полигональную сетку интерполируя только по глубине и заливает одним цветом
def drawMeshShaded(buffer:frameBuffer, mesh:meshData, mat:material, cam:camera = None):
    # направление освещения совпадает с направлением взгляда камеры
    if cam == None: cam = camera(); cam.lookAt(mesh.minWorldSpace, mesh.maxWorldSpace * 1.5);

    forward = cam.front();
    
    for f in mesh.faces:
        # переводим нормали вершин в мировое пространство
        n1 = (mesh.getNormalWorldSpace(f.n_1));
        n2 = (mesh.getNormalWorldSpace(f.n_2));
        n3 = (mesh.getNormalWorldSpace(f.n_3));

        a =  -MathUtils.dot(n1, forward);
        b =  -MathUtils.dot(n2, forward);
        c =  -MathUtils.dot(n3, forward);
        col = (MathUtils.clamp(0.0, 1.0, a) + MathUtils.clamp(0.0, 1.0, b) + MathUtils.clamp(0.0, 1.0, c)) * 0.333333333;
        # если скалярное произведение для всех нормалей на направление взгляда < 0, то не рисуем, тк
        # треугольник к нам задом(back-face culling)
        if col < 0: return;
         # перевоим точки в простраснтво отсечений камеры
        v1 = cam.toClipSpace(mesh.getVertWorldSpace(f.p_1));
        v2 = cam.toClipSpace(mesh.getVertWorldSpace(f.p_2));
        v3 = cam.toClipSpace(mesh.getVertWorldSpace(f.p_3));

        uv1 = mesh.uvs[f.uv1];
        uv2 = mesh.uvs[f.uv2];
        uv3 = mesh.uvs[f.uv3];
        
        drawTriangleShaded(buffer, vertex(pointToScrSpace(buffer, v1), n1, uv1),
                                   vertex(pointToScrSpace(buffer, v2), n2, uv2),
                                   vertex(pointToScrSpace(buffer, v3), n3, uv3),
                                   mat);