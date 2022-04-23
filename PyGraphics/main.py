import numpy       as np
import Transform
from   Camera      import camera;
from   MeshData    import meshData;
import MeshData    ;
from   MathUtils   import mat4,vec3,vec2;
from   FrameBuffer import RGB,frameBuffer;
import Graphics    as gr
from   Material    import  material;
import time
import lab_4;
from   PIL         import Image;
if __name__ == '__main__':
	fb = frameBuffer(1000,1000);
	fb.clearColor(RGB(255,255,255))
	mesh = meshData(); mesh.read("rabbit.obj");
	#mesh.read("fox_unify_normals.obj");
	mmat = material();
	mmat.setDiff("checkerboard-rainbow_.jpg");
	mmat.diffuse.tile = vec2(5,5);
	cam = camera();
	cam.lookAt(mesh.minLocalSpace,mesh.maxLocalSpace * 1.5);
	plane = MeshData.createPlane();
	plane.transformation.sx = 0.1;
	plane.transformation.sz = 0.1;
	#gr.drawMeshSolidInteractive(fb, mesh,cam);
	gr.drawMeshShaded(fb, plane, mmat, cam);
	gr.drawMeshSolidColor(fb, mesh,  cam); 
	fb.imshow();
	#gr.drawMeshShaded     (fb, mesh, mmat); 

	#gr.drawVertices       (fb,mesh);
	#gr.drawEdges          (fb,mesh);
	#mmat.diffuse.show();
	