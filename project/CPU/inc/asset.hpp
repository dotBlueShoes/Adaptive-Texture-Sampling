// Made by Matthew Strumillo 2024.07.20
//
#pragma once
#include <blue/types.hpp>

#define ATS_ASSET_WINDOW_TITLE      "ATS Window"
#define ATS_ASSET_BACKGROUND_COLOR  0.2f, 0.3f, 0.3f, 1.0f // BLUE-ISH COLOR
#define ATS_ASSET_OUTPUT_CANVAS_X   640
#define ATS_ASSET_OUTPUT_CANVAS_Y   640
#define ATS_ASSET_INPUT_CANVAS_X    320
#define ATS_ASSET_INPUT_CANVAS_Y    320

#define ATS_ASSET_RES_OUTPUT        "out\\"
#define ATS_ASSET_RES_TEXTURES      "res\\textures\\"
#define ATS_ASSET_RES_SHADERS       "res\\shaders\\"

namespace ASSET {

    // --- PNG'S

	const c8 FILE_OUT           [] = ATS_ASSET_RES_OUTPUT "output.png";
	//const c8 FILE_1_ALPHA       [] = ATS_ASSET_RES_TEXTURES "3_320x320x4.png";
	const c8 FILE_I0_NOALPHA     [] = ATS_ASSET_RES_TEXTURES "nearest\\" "0_320x320x3.png";
    const c8 FILE_I0_UPSCALE     [] = ATS_ASSET_RES_TEXTURES "upscale\\" "NEDI0_320x320x3.png";

    // ---


    // --- JPG'S
	const c8 FILE3 [] = ATS_ASSET_RES_TEXTURES "3.jpg";
    // ---


    // --- SHADERS
    const c8 SHADER_VERT_CUBE           [] = ATS_ASSET_RES_SHADERS "world\\cube.vert";
	const c8 SHADER_FRAG_CUBE           [] = ATS_ASSET_RES_SHADERS "world\\cube_nedi.frag";

	const c8 SHADER_VERT_FRAMEBUFFER    [] = ATS_ASSET_RES_SHADERS "screen\\framebuffer.vert";


    #if defined(ATS_ENABLE_FXAA_3_10)

        const c8 SHADER_FRAG_FRAMEBUFFER    [] = ATS_ASSET_RES_SHADERS "screen\\framebuffer_fxaa_3_10.frag";
    
    #elif defined(ATS_ENABLE_FXAA_3_11)

        const c8 SHADER_FRAG_FRAMEBUFFER    [] = ATS_ASSET_RES_SHADERS "screen\\framebuffer_fxaa_3_11.frag";

    #elif defined(ATS_ENABLE_FRAMEBUFFER_DEPTH_TEXTURE) && defined(__CUDACC__) == false

        const c8 SHADER_FRAG_FRAMEBUFFER    [] = ATS_ASSET_RES_SHADERS "screen\\framebuffer_depth.frag";

    #elif defined(ATS_MSAA_DISABLE_MANUAL_RESOLVE) || (ATS_MSAA_LEVEL == 0)

        #ifdef ATS_ENABLE_FRAMEBUFFER_ATS_USING_EDGE
            const c8 SHADER_FRAG_FRAMEBUFFER    [] = ATS_ASSET_RES_SHADERS "screen\\framebuffer_edge.frag";
        #else
            const c8 SHADER_FRAG_FRAMEBUFFER    [] = ATS_ASSET_RES_SHADERS "screen\\framebuffer.frag";
        #endif

    #else

        #ifdef ATS_ENABLE_FRAMEBUFFER_ATS_USING_EDGE
            #warning "NOT IMPLEMENTED"
        #else
            const c8 SHADER_FRAG_FRAMEBUFFER    [] = ATS_ASSET_RES_SHADERS "screen\\framebuffer_msaa_640.frag";
        #endif

    #endif


    const c8 SHADER_COMP    [] = ATS_ASSET_RES_SHADERS "compute\\basic.comp";
    
    // ---

}
