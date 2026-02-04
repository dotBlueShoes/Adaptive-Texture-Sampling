#version 460 core

//$

// [ Explicit interface matching from vertex ]
layout (location = 0) in vec4 vertexColor;
layout (location = 1) in vec2 vertexUVs;

// [ Uniform locations ]
layout (location = 0) uniform vec4 color;
layout (location = 1) uniform sampler2D tex;

// [ Fragment outputs ]
layout (location = 0) out vec4 finalColor;

void main() {
	finalColor = texture (tex, vertexUVs);
	//finalColor.a = 0;
}
