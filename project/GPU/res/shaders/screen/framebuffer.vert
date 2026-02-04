#version 460 core

//$

// [ Vertex attribute locations ]
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 uvs;

// [ Explicit interface matching to fragment ]
layout (location = 0) out vec2 vertexUVs;

void main() {
    gl_Position = vec4 (position.x, position.y, 0.0, 1.0); 
    vertexUVs = uvs;
}
