#version 460 core

//$

// --- FXAA
// Minimum threshold to prevent division by zero when calculating edge direction. min > 1/8, max > 1/128
const float fxaaReduceMin = 1.0 / 32.0f; // 1.0f / 64.0f;
// Multiplier to reduce the sensitivity to noise. min > 1/4, max > 1/8
const float fxaaReduceMul = 1.0 / 4.0f; // 1.0f / 16.0f;
// Maximum distance in pixels to sample along the edge. min-> 4  , max > 16
const float fxaaSpanMax = 4.0f; // 2.0f;
// ---


// --- ATS
const float texSize = 320;
const float upsSize = 640;
const float fvpSize = 640;
const vec3 grayscaleWeights = vec3 (0.299, 0.587, 0.114);
const float k = 3.0f;
// ---


// [ Explicit interface matching from vertex ]
layout (location = 0) in vec2 vertexUVs;

// [ Uniform locations ]
layout (location = 0) uniform sampler2D texDiffuse;
layout (location = 1) uniform sampler2D texUpscale;

// [ Fragment outputs ]
layout (location = 0) out vec4 finalColor;


vec3 GetTR (vec2 newUVs) {
    vec2 upUvs = vec2 (newUVs.x, newUVs.y - (1 / fvpSize));
    vec3 up = texture (texUpscale, upUvs).rgb;

    vec2 rtUvs = vec2 (newUVs.x + (1 / fvpSize), newUVs.y);
    vec3 rt = texture (texDiffuse, rtUvs).rgb;

    vec3 lt = texture (texDiffuse, newUVs).rgb;
    vec3 dn = texture (texUpscale, newUVs).rgb;

    float gup = dot (up, grayscaleWeights);
    float grt = dot (rt, grayscaleWeights);
    float glt = dot (lt, grayscaleWeights);
    float gdn = dot (dn, grayscaleWeights);

    vec3 color;

    float lc = (gup + grt + glt + gdn) / 4;
    float wl = exp(-abs(glt - lc) * k);
    float wr = exp(-abs(grt - lc) * k);
    float wu = exp(-abs(gup - lc) * k);
    float wd = exp(-abs(gdn - lc) * k);
    float wsum = wl + wr + wu + wd;
    color = (wl * lt + wr * rt + wu * up + wd * dn) / wsum;

    return color;
}


vec3 GetDL (vec2 newUVs) {
    vec3 up = texture (texUpscale, newUVs).rgb;
    vec3 rt = texture (texDiffuse, newUVs).rgb;

    vec2 dnUvs = vec2 (newUVs.x, newUVs.y + (1 / fvpSize));
    vec3 dn = texture (texDiffuse, dnUvs).rgb;

    vec2 ltUvs = vec2 (newUVs.x - (1 / fvpSize), newUVs.y);
    vec3 lt = texture (texUpscale, ltUvs).rgb;

    float gup = dot (up, grayscaleWeights);
    float grt = dot (rt, grayscaleWeights);
    float glt = dot (lt, grayscaleWeights);
    float gdn = dot (dn, grayscaleWeights);

    vec3 color;

    float lc = (gup + grt + glt + gdn) / 4;
    float wl = exp (-abs(glt - lc) * k);
    float wr = exp (-abs(grt - lc) * k);
    float wu = exp (-abs(gup - lc) * k);
    float wd = exp (-abs(gdn - lc) * k);
    float wsum = wl + wr + wu + wd;
    color = (wl * lt + wr * rt + wu * up + wd * dn) / wsum;

    return color;
}


vec3 GetIntertwinedFragment (vec2 newUVs) {
    vec2 pointUpscaledFraction = (gl_FragCoord.xy - vec2(0.5)); // 0 <-> (fvpSize - 1) // same
    // --- Flip Y (gl_FragCoord.xy goes from bot to top in Y axis)
    pointUpscaledFraction.y = 640 - 1 - pointUpscaledFraction.y;
    ivec2 tile = ivec2 (pointUpscaledFraction) & 1;
    vec3 fcolor;
    
    if        (tile.x == 0 && tile.y == 0) { // LU
        fcolor = texture (texUpscale, newUVs).rgb; 
    } else if (tile.x == 1 && tile.y == 0) { // UR
        fcolor = GetTR (newUVs);
    } else if (tile.x == 0 && tile.y == 1) { // DL
        fcolor = GetDL (newUVs);
    } else {                                 // RD
        fcolor = texture (texDiffuse, newUVs).rgb; 
    }

    return fcolor;
}


void main () {

    // --- SIMPLE NEAREST
    //finalColor = texture (texDiffuse, vertexUVs);
    // ---

    // --- CURRENT
    //vec3 fcolor = GetIntertwinedFragment (vertexUVs);
    //finalColor = vec4 (fcolor.r, fcolor.g, fcolor.b, 1.0f);
    // ---

    const vec2 framebufferSize = vec2 (640, 640);

    // --- Correct sizes
    vec2 inverseSourceTexSize = 1.0 / framebufferSize;

    // --- FXAA grayscale
    const vec3 grayscale = vec3(0.299, 0.587, 0.114);

    // --- Neighbor luminance in SOURCE texture space
    float gslt = dot(grayscale, GetIntertwinedFragment(vertexUVs + vec2(-1.0, -1.0) * inverseSourceTexSize));
    float gstr = dot(grayscale, GetIntertwinedFragment(vertexUVs + vec2( 1.0, -1.0) * inverseSourceTexSize));
    float gsrb = dot(grayscale, GetIntertwinedFragment(vertexUVs + vec2( 1.0,  1.0) * inverseSourceTexSize));
    float gsbl = dot(grayscale, GetIntertwinedFragment(vertexUVs + vec2(-1.0,  1.0) * inverseSourceTexSize));
    float gsmm = dot(grayscale, GetIntertwinedFragment(vertexUVs).rgb);

    // --- Edge direction
    vec2 dir;
    dir.x = -((gslt + gstr) - (gsbl + gsrb));
    dir.y =  ((gslt + gsbl) - (gstr + gsrb));

    const float REDUCE_SCALE = 0.33;
    float dirReduce = max((gslt + gstr + gsbl + gsrb) * (fxaaReduceMul * REDUCE_SCALE), fxaaReduceMin);

    float inverseDirAdjustment = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);

    // --- Direction scaled in SOURCE texel units
    dir = clamp(dir * inverseDirAdjustment, -fxaaSpanMax, fxaaSpanMax) * inverseSourceTexSize;

    // --- FXAA sampling along edge
    vec3 result1 =
        0.375 * GetIntertwinedFragment(vertexUVs + dir * (1.0/3.0 - 0.5)) +
        0.625 * GetIntertwinedFragment(vertexUVs + dir * (2.0/3.0 - 0.5));

    vec3 result2 = result1 * 0.5 + 0.25 * (
        GetIntertwinedFragment(vertexUVs + dir * (0.0/3.0 - 0.5)) +
        GetIntertwinedFragment(vertexUVs + dir * (3.0/3.0 - 0.5))
    );

    // --- Luma range check
    float lumaMin = min(gsmm, min(min(gslt, gstr), min(gsbl, gsrb))) + dirReduce;
    float lumaMax = max(gsmm, max(max(gslt, gstr), max(gsbl, gsrb))) - dirReduce;
    float lumaResult = dot(grayscale, result2);

    // --- Final FXAA output
    if (lumaResult < lumaMin || lumaResult > lumaMax)
        finalColor = vec4(result1, 1.0);
    else
        finalColor = vec4(result2, 1.0);

}
