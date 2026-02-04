#version 460 core

//$

//  IMPORTANT ! 
// this only works with MSAA set to 4!

//  ISSUE
// This is not really an MSAA because it used in shader... it's SSAA
// Real MSAA only happens during coverage test.

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

    //  My GPU sampling patterns resolves to a 
    // diamond pattern like:
    //
    // 0: x 0.375000, y 0.125000
    // 1: x 0.875000, y 0.375000
    // 2: x 0.125000, y 0.625000
    // 3: x 0.625000, y 0.875000

    // --- Translate point in [640] to point in [320] space.
    ivec2 size = textureSize(tex);
    ivec2 screenCoord = ivec2(vertexUVs * vec2(size * 2));
    ivec2 texelCoord = screenCoord / 2;

    // --- Local position inside the 2x2 block.
    ivec2 local = screenCoord % 2;
    int subsample = local.x + 2 * local.y; // (must match the packing)

    // --- Get subpixel from msaa buffer using subsamples.
    vec4 color = texelFetch (tex, texelCoord, subsample);

    finalColor = color;
}
