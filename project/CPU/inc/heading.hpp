#pragma once

/// #define ATS_DISABLE_INTEGRATED_GPU_GUARD 

#ifndef ATS_DISABLE_INTEGRATED_GPU_GUARD

    //  WINDOWS-ONLY-HINT 
    // Ensure use of a graphics card. Without it integrated card could be used.
    //
    extern "C" {
        __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
    }

#endif

//
// BLUE-LIB compilation params
//
#define CONSOLE_COLOR_ENABLED
#define LOGGER_TIME_FORMAT "%f"
#define MEMORY_TYPE_NOT_SIZED

// 
// ATS Constants
// 
#include "constants.hpp"

// 
// Compile-Time Settings. Should be passed via build script.
// 
#include "settings.hpp"

//
// HELPERS
//
#define ATS_SHADER_VERT ATS_MAX_CHARS_FOR_SHADER_VERT_FILE, "vertex"
#define ATS_SHADER_FRAG ATS_MAX_CHARS_FOR_SHADER_FRAG_FILE, "fragment"
#define ATS_SHADER_COMP ATS_MAX_CHARS_FOR_SHADER_COMP_FILE, "compute"
