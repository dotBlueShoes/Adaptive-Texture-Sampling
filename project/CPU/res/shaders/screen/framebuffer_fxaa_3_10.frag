#version 460 core

//$

// Minimum threshold to prevent division by zero when calculating edge direction.
// min -> 1/8, max -> 1/128
const float fxaaReduceMin = 1.0 / 128.0f;
// Multiplier to reduce the sensitivity to noise.
// min -> 1/4, max -> 1/8
const float fxaaReduceMul = 1.0 / 8.0f;
// Maximum distance in pixels to sample along the edge.
// min -> 4, max -> 16
const float fxaaSpanMax = 4.0f;

// [ Explicit interface matching from vertex ]
layout (location = 0) in vec2 vertexUVs;

// [ Uniform locations ]
layout (location = 0) uniform sampler2D tex;

// [ Fragment outputs ]
layout (location = 0) out vec4 finalColor;

void main() {

    //  Theres 2 problems
    // - Upsampling makes it all weird. (holes + wide aa like rim effect)
    // - Corners are blinking

    //  ABOUT
    // The compiler can resolve it to a uniform constant, so it’s basically free to 
    //  call textureSize() func instead of passing that value with a uniform.
    //
    // `inverseViewportSize` -> Suppose you want to sample 1 pixel to the right of 
    //   the current fragment. In pixel coordinates, that’s just +1. But in UV coordinates, 
    //   we'll need to scale it to [0-1] space. So
    // `vertexUVs + vec2(1.0, 0.0) * inverseViewportSize`
    //   is moving exactly 1 pixel to the right in UV space.
    //
    ivec2 viewportSize = textureSize (tex, 0);
    vec2 inverseViewportSize = vec2(1.0f / viewportSize.x, 1.0f / viewportSize.y);

    vec2 vertexUVs = gl_FragCoord.xy / viewportSize;

    const vec3 grayscale = vec3 (0.299, 0.587, 0.114);

	float gslt = dot (grayscale, texture (tex, vertexUVs + (vec2 (-1.0, -1.0) * inverseViewportSize)).rgb);
	float gstr = dot (grayscale, texture (tex, vertexUVs + (vec2 ( 1.0, -1.0) * inverseViewportSize)).rgb);
    float gsrb = dot (grayscale, texture (tex, vertexUVs + (vec2 ( 1.0,  1.0) * inverseViewportSize)).rgb);
	float gsbl = dot (grayscale, texture (tex, vertexUVs + (vec2 (-1.0,  1.0) * inverseViewportSize)).rgb);
	float gsmm = dot (grayscale, texture (tex, vertexUVs).rgb);

    //  ABOUT
    // Determine edge direction and whether or not it is an edge. 
    //  if 0 on x and 0 on y -> no edge detected.
    //
    vec2 dir;
	dir.x = -((gslt + gstr) - (gsbl + gsrb));
	dir.y =  ((gslt + gsbl) - (gstr + gsrb));

    //finalColor = vec4 (dir.x, dir.x, dir.x, 1.0);
    //finalColor = vec4 (dir.y, dir.y, dir.y, 1.0);

    //
    // Determine how far to sample along the edge for blending.
    //

    // 'dirReduce' ensures small luminance differences don’t blow up the gradient.
    //
    const float REDUCE_SCALE = 0.25;
    float dirReduce = max ((gslt + gstr + gsbl + gsrb) * (fxaaReduceMul * REDUCE_SCALE), fxaaReduceMin);

    // `inverseDirAdjustment` is normalized vector.
    //
	float inverseDirAdjustment = 1.0 / (min (abs (dir.x), abs (dir.y)) + dirReduce);

    // Clamp `dir` to `fxaaSpanMax` pixels to prevent overshooting.
    //  and multiply by `inverseViewportSize` to convert to UV space.
    //  So we get how long is dir vector in pixels!
    //
    dir = min (
        vec2 (fxaaSpanMax, fxaaSpanMax), 
        max (vec2 (-fxaaSpanMax, -fxaaSpanMax), dir * inverseDirAdjustment)
    ) * inverseViewportSize;


    //
    // Sample 4 points along the edge direction and blend them.
    //  This should produce a smooth interpolation along the edge.
    //


    // Conservative Smoothing
    // `result1` is the average of the middle two samples.
    //
    vec3 result1 = (1.0 / 2.0) * (
		texture (tex, vertexUVs + (dir * vec2 (1.0 / 3.0 - 0.5))).rgb +
		texture (tex, vertexUVs + (dir * vec2 (2.0 / 3.0 - 0.5))).rgb
    );

    // Strong Smoothing
    // `result2` adds the first and last samples with smaller weight.
    //
	vec3 result2 = result1 * (1.0 / 2.0) + (1.0 / 4.0) * (
		texture (tex, vertexUVs + (dir * vec2 (0.0 / 3.0 - 0.5))).rgb +
		texture (tex, vertexUVs + (dir * vec2 (3.0 / 3.0 - 0.5))).rgb
    );

    // Compute min/max luminance of the 5 sampled pixels.
    //  `lumaResult` is the luminance of the blended result.
    //
	float lumaMin = min (gsmm, min (min (gslt, gstr), min (gsbl, gsrb)));
	float lumaMax = max (gsmm, max (max (gslt, gstr), max (gsbl, gsrb)));
	float lumaResult = dot (grayscale, result2);

    // If result2 overshoots luminance bounds, fallback to result1.
    //  this should prevent artifacts like dark/light halos along edges.
    //
    if (lumaResult < lumaMin || lumaResult > lumaMax)
		finalColor = vec4 (result1, 1.0);
	else
		finalColor = vec4 (result2, 1.0);


    //finalColor = texture (tex, vertexUVs);
}