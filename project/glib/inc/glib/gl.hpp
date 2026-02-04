// Made by Matthew Strumillo 2024.07.20
//
#pragma once
//
#include <blue/error.hpp>
#include <glad/glad.h>
//
#include <GLFW/glfw3.h>
//
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

//#include "tool/debug.hpp"

// constexpr auto GetGLSLVersion() {
// 	#if defined(IMGUI_IMPL_OPENGL_ES2) 	// GLES 2.0 + GLSL 100
// 		return "#version 100";
// 	#elif defined(__APPLE__) 			// GLES 3.2 + GLSL 150
// 		return "#version 150";
// 	#else 								// GLES 3.0 + GLSL 130
// 		return "#version 130";
// 	#endif
// }

namespace GLM {

    //ATS_TRANSFORM_ROTATION_Z, ATS_TRANSFORM_ROTATION_Y, ATS_TRANSFORM_ROTATION_X

    void RotateByDegrees (
        glm::mat4& transform,
        const r32& x,
        const r32& y,
        const r32& z
    ) {
        #if ATS_COORD_SYS == ATS_COORD_SYS_ZYX
            transform = glm::rotate (
                transform, glm::radians (z), glm::vec3 (0.0f, 0.0f, -1.0f)
            ); 

            transform = glm::rotate (
                transform, glm::radians (y), glm::vec3 (0.0f, 1.0f, 0.0f)
            ); 

            transform = glm::rotate (
                transform, glm::radians (x), glm::vec3 (-1.0f, 0.0f, 0.0f)
            ); 
        #elif ATS_COORD_SYS == ATS_COORD_SYS_XYZ
            transform = glm::rotate (
                transform, glm::radians (x), glm::vec3 (1.0f, 0.0f, 0.0f)
            ); 

            transform = glm::rotate (
                transform, glm::radians (y), glm::vec3 (0.0f, 1.0f, 0.0f)
            ); 

            transform = glm::rotate (
                transform, glm::radians (z), glm::vec3 (0.0f, 0.0f, 1.0f)
            );
        #endif
    }

}

namespace GLM::VIEW {

	#define GLMPTR(data) glm::value_ptr(data)
	
	// PERSPECTIVE
	auto Perspective (r32 fov, r32 ratio) {
		return glm::perspective (
			glm::radians (fov), 
			ratio,
			ATS_DEPTH_NEAR, ATS_DEPTH_FAR
		);
	}

}


#define GL_GET_ERROR(msg, args) { \
    GLenum glerror = glGetError (); \
	if (glerror != GL_NO_ERROR) LOGERROR ("GL: [%d] at (" msg ")\n", glerror, args); \
}


namespace GL {

	// > [0000] 'GL_NO_ERROR'
	// No error has been recorded. The value of this symbolic constant is guaranteed to be 0.
	// > [1280] 'GL_INVALID_ENUM'
	// An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag.
	// > [1281] 'GL_INVALID_VALUE'
	// A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag.
	// > [1282] 'GL_INVALID_OPERATION'
	// The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag.
	// > [1283] 'GL_STACK_OVERFLOW'
	// An attempt has been made to perform an operation that would cause an internal stack to overflow.
	// > [1284] 'GL_STACK_UNDERFLOW'
	// An attempt has been made to perform an operation that would cause an internal stack to underflow.
	// > [1285] 'GL_OUT_OF_MEMORY'
	// There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded.
	// > [1286] 'GL_INVALID_FRAMEBUFFER_OPERATION'
	// The framebuffer object is not complete. The offending command is ignored and has no other side effect than to set the error flag.

	// GL_NO_ERROR
	// GL_INVALID_ENUM
	// GL_INVALID_VALUE
	// GL_INVALID_OPERATION
	// GL_INVALID_FRAMEBUFFER_OPERATION
	// GL_OUT_OF_MEMORY
	// GL_STACK_UNDERFLOW
	// GL_STACK_OVERFLOW

	void GetError (const c8* const& msg) {
		GLenum error = glGetError ();
		if (error != GL_NO_ERROR) LOGERROR ("GL: [%d] at (%s)\n", error, msg);
	}

	void GLLogFrameBufferError (const u32& framebufferId) {
		GLenum framebufferStatus = glCheckFramebufferStatus (GL_FRAMEBUFFER);

    	switch (framebufferStatus) {

    	    case GL_FRAMEBUFFER_COMPLETE: 
				LOGINFO ("Framebuffer Object {0} is Fine.", framebufferId);
				break;

    	    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
    	        LOGERROR ("Framebuffer Object {0} Error: Attachment Point Unconnected", framebufferId);
    	        break;

    	    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
    	        LOGERROR ("Framebuffer Object {0} Error: Missing Attachment", framebufferId);
    	        break;

    	    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
    	        LOGERROR ("Framebuffer Object {0} Error: Draw Buffer", framebufferId);
    	        break;

    	    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
    	        LOGERROR ("Framebuffer Object {0} Error: Read Buffer", framebufferId);
    	        break;

    	    case GL_FRAMEBUFFER_UNSUPPORTED:
    	        LOGERROR ("Framebuffer Object {0} Error: Unsupported Framebuffer Configuration", framebufferId);
    	        break;

    	    default:
    	        LOGERROR ("Framebuffer Object {0} Error: Unknown Framebuffer Object Failure", framebufferId);
    	        break;

    	}
	}


    auto cmpstr_a (
        const c8* const& str1,
        const c8* const& str2
    ) {
        u16 condition = 0;
        u32 i = 0;

        for (; str1[i] != '\0' || str2[i] != '\0'; ++i) {
            condition += (str1[i] != str2[i]);
        }

        // Ensure length equality
        condition += !(str1[i] == '\0' && str2[i] == '\0');

        return condition;
    }


    void CheckExtension (
        u32 const& extensionsCount,
        const c8* const& extensionName
    ) {
        for (u32 j = 0; j < extensionsCount; ++j) {
            auto&& extension = (const c8* const) glGetStringi (GL_EXTENSIONS, j);

            if (cmpstr_a (extensionName, extension) == 0) { 
                LOGINFO ("Extension `%s` is available.\n", extensionName);
                return;
            }
        }

        LOGERROR ("Extension `%s` is not available.\n", extensionName);
    }

    void FunctionalityCheck () {
        GLint extensionsCount = 0;

        glGetIntegerv (GL_NUM_EXTENSIONS, &extensionsCount);
        LOGINFO ("extensions: %d\n", extensionsCount); 

        { // Extension availability
            CheckExtension (extensionsCount, ATS_GL_EXTENSION_TEXTURE_FILTER_ANISOTROPIC);
            CheckExtension (extensionsCount, ATS_GL_EXTENSION_SPIR_V_SHADER_BINARIES);
        }
    }

}
