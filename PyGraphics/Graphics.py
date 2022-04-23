import numpy       as np
import MathUtils;
from   MathUtils   import vec2,vec3,mat4
from   Material    import material;
import Camera      
from   Camera      import camera
from   FrameBuffer import frameBuffer
from   FrameBuffer import RGB
from   MeshData    import meshData;
import tkinter as tk
from   PIL import ImageTk, Image




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

def lerpVertex(a:vertex ,b:vertex ,val:float)->vertex:
    return a + (b - a) * val;

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
                buffer.setPixel(y, x, color)
            else:
                buffer.setPixel(x, y, color)
            error = error + derror
            if error > 0.5:
                y += 1 if y1 > y0 else -1
                error -= 1
# рисование линии, четвертый вариант алгоритма (алгоримтм Брезенхема)
def drawLineV5(buffer:frameBuffer, x0: int, y0: int, depth0: float,
                                   x1: int, y1: int, depth1: float,
                                   color: RGB = RGB(255, 255, 255)):
        steep = False
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            depth0, depth1 = depth1,depth0;
            x0, x1 = x1, x0;
            y0, y1 = y1, y0;
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0:return;
        derror = abs(dy / float(dx))
        error = 0.0
        y = y0
        ddepth = (depth1 - depth0) / (x1 - x0);
        cdepth = depth0 - ddepth;
        for x in range(int(x0), int(x1)):
            cdepth += ddepth
            if steep:
               if buffer.setDepth(y,x,cdepth):buffer.setPixel(y, x, color);
            else:
               if buffer.setDepth(x,y,cdepth):buffer.setPixel(x, y, color)
            error = error + derror
            if error > 0.5:
                y += 1 if y1 > y0 else -1
                error -= 1

def drawPoint(buffer:frameBuffer, x:int, y:int, color:RGB = RGB(255, 255, 255), depth:float = 0):
       if not buffer.setDepth(x,y,depth):
           return;
       buffer.setPixel(x - 1, y - 1, color);
       buffer.setPixel(x - 1, y,     color);
       buffer.setPixel(x - 1, y + 1, color);
       buffer.setPixel(x,     y - 1, color);
       buffer.setPixel(x,     y,     color);
       buffer.setPixel(x,     y + 1, color);
       buffer.setPixel(x + 1, y - 1, color);
       buffer.setPixel(x + 1, y,     color);
       buffer.setPixel(x + 1, y + 1, color);

def pointToScrSpace(buffer:frameBuffer,pt:vec3)->vec3:
    return vec3(round(MathUtils.clamp(0, buffer.width -1, round(buffer.width  * ( pt.x * 0.5 + 0.5)))),
                round(MathUtils.clamp(0, buffer.height-1, round(buffer.height * (-pt.y * 0.5 + 0.5)))),
                pt.z);

#отрисовка одноцветного треугольника(интерполируется только глубина) 
def drawTriangleSolid(buffer:frameBuffer, p0:vertex, p1:vertex, p2:vertex, color:RGB = RGB(255, 0, 0)):
    if p0.v.y == p1.v.y and p0.v.y == p2.v.y:return; #i dont care about degenerate triangles
    # sort the vertices, p0, p1, p2 lower-to-upper (bubblesort yay!)
    if (p0.v.y > p1.v.y): p0, p1 = p1, p0;
    if (p0.v.y > p2.v.y): p0, p2 = p2, p0;
    if (p1.v.y > p2.v.y): p1, p2 = p2, p1;
    
    total_height:int = round(p2.v.y - p0.v.y);
    
    for i in range( 0, total_height):
         second_half:bool = i > p1.v.y-p0.v.y or p1.v.y == p0.v.y;
         
         segment_height:int = p1.v.y-p0.v.y;
         
         if second_half:segment_height:int = p2.v.y-p1.v.y;
        
         if segment_height==0:continue;

         alpha:float  = float(i)/total_height;

         beta:float   = 0;

         if second_half: beta = float(i - (p1.v.y - p0.v.y))/segment_height;
         else:beta = float(i / segment_height);# be careful: with above conditions no division by zero her

         A = lerpVertex(p0, p2, alpha);
         
         if second_half: B = lerpVertex(p1, p2, beta);
         else:B = lerpVertex(p0, p1, beta);
         
         if (A.v.x > B.v.x): A, B = B, A;
         for j in range(round(A.v.x), round(B.v.x)):
             phi:float = 0.0;
             if B.v.x == A.v.x: phi = 1.0
             else: phi = float(j - A.v.x) / float(B.v.x - A.v.x);
             P:vec3 = lerpVertex(A, B, phi);
             zx, xy = round(P.v.x), round(P.v.y);
             if buffer.setDepth(zx, xy, P.v.z): 
                colShading:float = MathUtils.clamp(0.0, 1.0, MathUtils.dot(P.n, vec3(0.333, 0.333, 0.333)));
                buffer.setPixel(zx, xy, RGB(color.R * colShading, color.G * colShading, color.B * colShading));

#отрисовка треугольника(интерполируется только глубина, нормали, барицентрические координаты) 
def drawTriangleShaded(buffer:frameBuffer, p0:vertex, p1:vertex, p2:vertex, mat:material): #позиции(в прострастве экрана) вершин треугольника
    if p0.v.y == p1.v.y and p0.v.y == p2.v.y:return; #i dont care about degenerate triangles
    # sort the vertices, p0, p1, p2 lower-to-upper (bubblesort yay!)
    if (p0.v.y > p1.v.y): p0, p1   = p1, p0;

    if (p0.v.y > p2.v.y): p0, p2   = p2, p0;

    if (p1.v.y > p2.v.y): p1, p2   = p2, p1;
    
    total_height:int = round(p2.v.y - p0.v.y);
    
    for i in range(0, total_height):
         second_half:bool = i > p1.v.y-p0.v.y or p1.v.y == p0.v.y;
         
         segment_height:int = p1.v.y-p0.v.y;
         
         if second_half:segment_height:int = p2.v.y-p1.v.y;
        
         if segment_height==0:continue;

         alpha:float  = float(i)/total_height;

         beta:float   = 0;

         if second_half: beta = float(i - (p1.v.y - p0.v.y))/segment_height;
         else:beta = float(i / segment_height)# be careful: with above conditions no division by zero her
         A   = lerpVertex(p0, p2, alpha);
         if second_half: B = lerpVertex(p1, p2, beta);
         else: B = lerpVertex(p0, p1,beta);
         if (A.v.x > B.v.x): A, B = B, A;

         for j in range(round(A.v.x), round(B.v.x)):
             phi:float = 0.0;
             if B.v.x == A.v.x: phi = 1.0
             else: phi = float(j - A.v.x) / float(B.v.x - A.v.x);
             P   = lerpVertex(A, B, phi);
             ix, jy = round(P.v.x), round(P.v.y);
             if buffer.setDepth(ix, jy, P.v.z): 
                col:RGB = mat.diffColor(P.uv);
                colShading:float = MathUtils.clamp(0.0, 1.0, MathUtils.dot(P.n, vec3(0.333,0.333,0.333)));
                buffer.setPixel(ix, jy, RGB(col.R * colShading, col.G * colShading, col.B * colShading));

# отрисовка вершин
def drawVertices(buffer:frameBuffer, mesh:meshData, cam:camera = None, color:RGB = RGB(0, 0, 255)):
        if cam == None:cam = Camera.renderCamera(buffer, mesh.minWorldSpace, mesh.maxWorldSpace * 1.5);
        for point in mesh.vertices:
            v1 = pointToScrSpace(buffer, cam.toClipSpace(mesh.transformation.transformVect(point,1)));
            # if buffer.zBuffer[v1.x,v1.y] > v1.z:return;
            drawPoint(buffer,v1.x, v1.y, color, v1.z);

# отрисовка ребер
def drawEdges(buffer:frameBuffer, mesh:meshData, cam:camera = None, color:RGB = RGB(0, 0, 0)):
        if cam == None:cam = camera();cam.lookAt(mesh.minWorldSpace, mesh.maxWorldSpace * 1.5);
        # направление освещения совпадает с направлением взгляда камеры
        forward = cam.front;
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

            if a > 0 or b > 0:drawLineV4(buffer, v1.x, v1.y, v2.x, v2.y, color);
            if a > 0 or c > 0:drawLineV4(buffer, v1.x, v1.y, v3.x, v3.y, color);
            if b > 0 or c > 0:drawLineV4(buffer, v2.x, v2.y, v3.x, v3.y, color);

import threading
import time
#ГУЙ
debugWindow = None;
debugWindowlabel = None;

def createImageWinodow(fb:frameBuffer):
    global debugWindow;
    global debugWindowlabel;
    if debugWindow != None: return;
    debugWindow = tk.Tk()
    debugWindow.title("Image Viewer");
    img =  ImageTk.PhotoImage(fb.frameBufferImage);
    debugWindow.geometry(str(img.height() + 3)+ "x" + str(img.width() + 3))
    debugWindowlabel = tk.Label(image = img)
    debugWindowlabel.pack(side="bottom", fill="both", expand="yes")
    while 'normal' == debugWindow.state():
       try:
            debugWindow.update();
            updateImageWindow(fb);
       except Exception:print("GUI exeqution stops")

def updateImageWindow(fb:frameBuffer):
    if debugWindow == None:return;
    img =  ImageTk.PhotoImage(fb.frameBufferImage);
    debugWindowlabel.configure(image = img);
    debugWindowlabel.image = img;

# рисует полигональную сетку интерполируя только по глубине и заливает одним цветом
def drawMeshSolidColor(buffer:frameBuffer, mesh:meshData, cam:camera = None, color:RGB = RGB(255, 200, 125)):
    # направление освещения совпадает с направлением взгляда камеры
    if cam == None: cam = Camera.renderCamera(buffer, mesh.minWorldSpace, mesh.maxWorldSpace * 1.5);
    forward = cam.front;
    uv = vec2(0,0);
    for f in mesh.faces:
        # переводим нормали вершин в мировое пространство
        n1 = mesh.getNormalWorldSpace(f.n_1);
        n2 = mesh.getNormalWorldSpace(f.n_2);
        n3 = mesh.getNormalWorldSpace(f.n_3);
        # треугольник к нам задом(back-face culling)
        if MathUtils.dot(n1, forward) > 0 and MathUtils.dot(n2, forward) > 0 and MathUtils.dot(n3, forward) > 0 : continue;
        # перевоим точки в простраснтво отсечений камеры
        v1 = cam.toClipSpace(mesh.getVertWorldSpace(f.p_1));
        v2 = cam.toClipSpace(mesh.getVertWorldSpace(f.p_2));
        v3 = cam.toClipSpace(mesh.getVertWorldSpace(f.p_3));
        drawTriangleSolid(buffer,vertex(pointToScrSpace(buffer, v1), n1, uv),
                                 vertex(pointToScrSpace(buffer, v2), n2, uv),
                                 vertex(pointToScrSpace(buffer, v3), n3, uv),
                                 color);

def drawMeshSolidInteractive(buffer:frameBuffer, mesh:meshData, cam:camera = None, color:RGB = RGB(255, 200, 125)):
    rendererThread = threading.Thread(target = drawMeshSolidColor, args=(buffer, mesh, cam, ))
    rendererThread.start();
    createImageWinodow(buffer);

# рисует полигональную сетку интерполируя только по глубине и заливает одним цветом
def drawMeshShaded(buffer:frameBuffer, mesh:meshData, mat:material, cam:camera = None):
    # направление освещения совпадает с направлением взгляда камеры
    if cam == None: cam = Camera.renderCamera(buffer, mesh.minWorldSpace, mesh.maxWorldSpace * 1.5);
    
    forward = cam.front;
    for f in mesh.faces:
        # переводим нормали вершин в мировое пространство
        n1 = (mesh.getNormalWorldSpace(f.n_1));
        n2 = (mesh.getNormalWorldSpace(f.n_2));
        n3 = (mesh.getNormalWorldSpace(f.n_3));
        # треугольник к нам задом(back-face culling)
        if MathUtils.dot(n1, forward) > 0 and MathUtils.dot(n2, forward) > 0 and MathUtils.dot(n3, forward) > 0 : continue;
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


def drawMeshShadedInteractive(buffer:frameBuffer, mesh:meshData, mat:material, cam:camera = None):
    rendererThread = threading.Thread(target = drawMeshShaded, args=(buffer, mesh, mat, cam, ))
    rendererThread.start();
    createImageWinodow(buffer);