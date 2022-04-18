import numpy       as np
import Transform
from   Camera      import camera;
from   MeshData    import meshData;
from   MathUtils   import mat4,vec3,vec2;
from   FrameBuffer import RGB,frameBuffer;
import Graphics    as gr
from   Material    import  material;
import time
import lab_4;
if __name__ == '__main__':
	
	fb = frameBuffer(1024,1024);
	fb.clearColor(RGB(255,255,255))
	mesh = meshData(); mesh.read("rabbit.obj");
	#mesh.read("fox_unify_normals.obj");
	mmat = material();
	mmat.setDiff("checkerboard-rainbow_.jpg");
	mmat.diffuse.tile = vec2(10,10);

	#gr.drawMeshSolidColor(fb, mesh); 
	gr.drawMeshShaded    (fb, mesh, mmat); 
	#gr.drawVertices      (fb,mesh);
	#gr.drawEdges         (fb,mesh);
	fb.imshow();
