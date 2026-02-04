#pragma once
#include "gl.hpp"

// Stringize helpers
#define GLSL_STR_NOEXPAND(x) #x
#define GLSL_STR(x) GLSL_STR_NOEXPAND(x)


// NOTE. This guard 0 is actually necceserry.
#if (ATS_MSAA_LEVEL != 0)
    #define GLSL_INJECTION_DEFINE_MSAA R"(#define MSAA_SAMPLES_LEVEL )" GLSL_STR(ATS_MSAA_LEVEL)
#else
    #define GLSL_INJECTION_DEFINE_MSAA R"(#define MSAA_SAMPLES_LEVEL 0)"
#endif


#define GLSL_INJECTION_DEFINE_INJECTION R"(#define ATS_GLSL_INJECTION)"

#define GLSL_INJECTION_INCLUDE_ALL \
    GLSL_INJECTION_DEFINE_INJECTION "\n" \
    GLSL_INJECTION_DEFINE_MSAA "\n"

#define GLSL_INJECTION_INCLUDE_ALL_LENGTH sizeof (GLSL_INJECTION_INCLUDE_ALL) - 1


//  NOTE
// We simply guess that all .vert & .frag files have a version header that ends at 19th character. 
//
#define GLSL_INJECTION_VERSION_OFFSET 19


namespace GLSL {

    c8* ffsd;

}
