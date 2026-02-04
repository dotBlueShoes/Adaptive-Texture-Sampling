#version 460 core

//$

// [ Vertex attribute locations ]
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uvs;

// [ Uniform locations ]
layout (location = 2) uniform mat4 projection;
layout (location = 3) uniform mat4 view;
layout (location = 4) uniform mat4 transform;

// [ Explicit interface matching to fragment ]
layout (location = 0) out vec4 vertexColor;
layout (location = 1) out vec2 vertexUVs;

void main() {
	gl_Position = projection * view * transform * vec4 (position, 1.0);
	vertexColor = vec4 (1.0, 0.0, 0.0, 1.0);
	vertexUVs = uvs;
}
