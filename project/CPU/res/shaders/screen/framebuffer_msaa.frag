#version 460 core

//$

#ifndef ATS_GLSL_INJECTION

    // Provided fallback values.
    #define MSAA_SAMPLES_LEVEL 4

#endif

// [ Explicit interface matching from vertex ]
layout (location = 0) in vec2 vertexUVs;

// [ Uniform locations ]
layout (location = 0) uniform sampler2DMS tex;

// [ Fragment outputs ]
layout (location = 0) out vec4 finalColor;

void main() {

    ivec2 texSize = textureSize (tex);
    ivec2 texCoord = ivec2 (vertexUVs * texSize);
    vec4 color = vec4 (0.0);

    //  ISSUE
    // This is not really an MSAA because it used in shader... it's SSAA
    // Real MSAA only happens during coverage test.

    for (int i = 0; i < MSAA_SAMPLES_LEVEL; i++) {
        color += texelFetch (tex, texCoord, i); // 0, 1, 2, 3
    }

    color /= float (MSAA_SAMPLES_LEVEL);

    finalColor = color;
}
