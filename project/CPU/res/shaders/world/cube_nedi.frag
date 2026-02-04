#version 460 core

//$

// [ Explicit interface matching from vertex ]
layout (location = 0) in vec4 vertexColor;
layout (location = 1) in vec2 vertexUVs;

// [ Uniform locations ]
layout (location = 0) uniform vec4 color;
layout (location = 1) uniform sampler2D texDiffuse;
layout (location = 5) uniform sampler2D texUpscale;

// [ Fragment outputs ]
layout (location = 0) out vec4 finalDiffuse;
layout (location = 1) out vec4 finalUpscale;

void main() {
    finalDiffuse = texture (texDiffuse, vertexUVs);
	finalUpscale = texture (texUpscale, vertexUVs);
}
