import numpy       as np
import Transform
from   Camera      import camera;
from   MeshData    import meshData;
from   MathUtils   import mat4,vec3,vec2;
from   FrameBuffer import RGB,frameBuffer;
import Graphics    as gr
from   Material    import  material;
import lab_4;
if __name__ == '__main__':
	frameBuffer = frameBuffer();
	frameBuffer.clearColor(RGB(255,255,255))
	mesh = meshData();
	mesh.read("rabbit.obj");
	#mesh.read("fox_unify_normals.obj");
	mmat = material();
	mmat.setDiff("checkerboard-rainbow_.jpg");
	mmat.diffuse.tile = vec2(10,10);

	gr.drawMeshSolidColor(frameBuffer, mesh); 
	#gr.drawMeshShaded(frameBuffer, mesh, mmat); 
	#gr.drawVertices  (frameBuffer,mesh);
	#gr.drawEdges     (frameBuffer,mesh);
	
	frameBuffer.imshow();
