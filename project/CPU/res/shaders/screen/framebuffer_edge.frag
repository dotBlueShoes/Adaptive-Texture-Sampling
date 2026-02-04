#version 460 core

//$

// Size of the edge.
const float offset = 1.0 / 300.0;

// [ Explicit interface matching from vertex ]
layout (location = 0) in vec2 vertexUVs;

// [ Uniform locations ]
layout (location = 0) uniform sampler2D tex;

// [ Fragment outputs ]
layout (location = 0) out vec4 finalColor;

void main() {

    vec2 offsets[9] = {
        vec2 (-offset,  offset), // top-left
        vec2 ( 0.0f,    offset), // top-center
        vec2 ( offset,  offset), // top-right
        vec2 (-offset,  0.0f),   // center-left
        vec2 ( 0.0f,    0.0f),   // center-center
        vec2 ( offset,  0.0f),   // center-right
        vec2 (-offset, -offset), // bottom-left
        vec2 ( 0.0f,   -offset), // bottom-center
        vec2 ( offset, -offset)  // bottom-right    
    };

    // Sharpening
    //float kernel[9] = {
    //    -1, -1, -1,
    //    -1,  9, -1,
    //    -1, -1, -1
    //};

    // Edge-detection
    float kernel[9] = {
        1,  1, 1,
        1, -8, 1,
        1,  1, 1
    };
    
    vec3 combinedTexture[9];

    for (int i = 0; i < 9; ++i) {
        combinedTexture[i] = vec3 (texture (tex, vertexUVs.xy + offsets[i]));
    }

    vec3 col = vec3(0.0);

    for (int i = 0; i < 9; ++i) col += combinedTexture[i] * kernel[i];
    
    finalColor = vec4 (col, 1.0);
}
