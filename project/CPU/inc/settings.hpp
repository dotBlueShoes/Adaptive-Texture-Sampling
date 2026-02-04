#pragma once

//  TODO
// Build script should pass all those values.
//

//  TODO
// Custom mipmaps -> #define ATS_ENABLE_CUSTOM_MIPMAP
//

// FLAGS

// Make the reder target a file instead.
#define ATS_ENABLE_RENDER_TO_FILE

// Prepare Framebuffer resources and if not ATS_ENABLE_RENDER_TO_FILE render it. 
//  On GPU profile means using CUDA INTERLOP (which requires a framebuffer) instead of `glReadPixels` function.
#define ATS_ENABLE_FRAMEBUFFER_RENDER // disable for CPU profile to work.

// Seek for used GL/CUDA extensions in GPU.
//#define ATS_ENABLE_EXTENSIONS_CHECK

// Seek for CUDA device properties like grid/block/warp size.
#define ATS_ENABLE_CUDA_DEVICE_CHECK

// Kernels, CUDA-OPENGL integrity.
//#define ATS_ENABLE_DEEP_DEBUG

// Enable code injection pre glsl shader compilation. 
#define ATS_GLSL_INJECTION

// Make an FPS counter in window title.
#define ATS_DISPLAY_FPS

// Make the object in scene rotate.
//#define ATS_ROTATE_OBJECT

// Enabling and setting to 0 disables V-Sync.
#define ATS_SWAP_INTERVAL 0


// Instead of storing depth and stencil inside a renderobject use a texture for depth only
//  This will allow us using the depth inside the shader.
//  ! CPU profile only !
//  ! overrides ATS_MSAA_LEVEL !
#define ATS_ENABLE_FRAMEBUFFER_DEPTH_TEXTURE

// Runs FXAA filter on the framebuffer.
//  ! CPU profile only !
//  ! overrides ATS_ENABLE_FRAMEBUFFER_DEPTH_TEXTURE !
//  ! overrides ATS_MSAA_LEVEL !
//#define ATS_ENABLE_FXAA_3_10
//#define ATS_ENABLE_FXAA_3_11
//#define ATS_ENABLE_CMAA2

// If any shader hits an error during compilation or linking stage
//  there will be an additional information present in the console about the error.
#define ATS_ENABLE_SHADER_ERROR_COMPILATION_LOGGING


// NUMERICS
#define ATS_ANISOTROPY_LEVEL                4 // 0 - means off.

//  ABOUT
// Real MSAA only happens during coverage check handling MSAA in fragment shader essentially makes it
//  work as SSAA. TODO. Right now `ATS_MSAA_DISABLE_MANUAL_RESOLVE` implementation is not complete.
//#define ATS_MSAA_DISABLE_MANUAL_RESOLVE
// ! CPU profile only ! - prob. missing logic for copying from multisample texture.
#define ATS_MSAA_LEVEL                      0 // Makes use of build-in msaa feature. 0 - means off.

#define ATS_MAX_CHARS_FOR_SHADER_VERT_FILE  1024
#define ATS_MAX_CHARS_FOR_SHADER_FRAG_FILE  4096 + 2048 + 2048 + 2048
#define ATS_MAX_CHARS_FOR_SHADER_COMP_FILE  1024
#define ATS_CAMERA_Z_SIGN                   +
#define ATS_CAMERA_FOV                      39.6f //39.5978f
//
//1 #define ATS_TRANSFORM_POSITION_X            1.0f
//1 #define ATS_TRANSFORM_POSITION_Y            0.0f
//1 #define ATS_TRANSFORM_POSITION_Z            0.0f
//1 #define ATS_TRANSFORM_ROTATION_X            0.0f
//1 #define ATS_TRANSFORM_ROTATION_Y            0.0f
//1 #define ATS_TRANSFORM_ROTATION_Z            0.0f
//
#define ATS_TRANSFORM_POSITION_X            0.0f
#define ATS_TRANSFORM_POSITION_Y            0.0f
#define ATS_TRANSFORM_POSITION_Z            0.0f
#define ATS_TRANSFORM_ROTATION_X            45.0f
#define ATS_TRANSFORM_ROTATION_Y            45.0f
#define ATS_TRANSFORM_ROTATION_Z            20.0f
#define ATS_TRANSFORM_SCALE_X               1.0f
#define ATS_TRANSFORM_SCALE_Y               1.0f
#define ATS_TRANSFORM_SCALE_Z               1.0f

// ! Remember to change these values in the shader code too !
//#define ATS_DEPTH_NEAR                      0.1f
//#define ATS_DEPTH_FAR                       100.0f
#define ATS_DEPTH_NEAR                      1.0f
#define ATS_DEPTH_FAR                       50.0f


// ENUMS
#define ATS_COORD_SYS                       ATS_COORD_SYS_ZYX
#define ATS_TEXTURE_MIPMAP_FILTERING_METHOD ATS_MIPMAP_LINEAR
//#define ATS_TEXTURE_MIPMAP_FILTERING_METHOD ATS_MIPMAP_NONE
#define ATS_TEXTURE_FILTERING_METHOD        ATS_NEAREST
// #define ATS_TEXTURE_FILTERING_METHOD     ATS_NEAREST
#define ATS_FRAMEBUFFER_FILTERING_METHOD    ATS_NEAREST
// #define ATS_FRAMEBUFFER_FILTERING_METHOD ATS_LINEAR
#define ATS_OUTPUT_FILTERING_METHOD         ATS_EDGE
