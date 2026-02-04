// Made by Matthew Strumillo 2024.03.08
//
#pragma once
#include "gl.hpp"
#include "uniform.hpp"
#include "glsl.hpp"

namespace SHADER::COMPUTE {

    void Compile (
        OUT		GLuint& 		sp,	 // shader-program
		IN		const c8* const csd  // compute-shader-data
    ) {

        #ifdef ATS_ENABLE_SHADER_ERROR_COMPILATION_LOGGING
            const auto SIZE = 1024;
    	    c8 errorBuffer[SIZE];
            s32 errorSize = 0;
		    GLint error;
        #endif

        GLuint cs = glCreateShader (GL_COMPUTE_SHADER);
        glShaderSource (cs, 1, &csd, NULL);
        glCompileShader (cs);

        #ifdef ATS_ENABLE_SHADER_ERROR_COMPILATION_LOGGING
            glGetShaderiv (cs, GL_COMPILE_STATUS, &error);
            if (!error) { 
    	        glGetShaderInfoLog (cs, SIZE, &errorSize, errorBuffer);
		    	LOGERROR ("ERROR::SHADER::COMPUTE::COMPILATION_FAILED\n");
                glDeleteShader (cs);
		    	ERROR ("MSG [%d]: \n%s\n", errorSize, errorBuffer);
    	    }
        #endif
                    
        sp = glCreateProgram ();
        glAttachShader (sp, cs);
        glLinkProgram (sp);

        // --- Deallocate handlers
        glDeleteShader (cs);
        // ---

        #ifdef ATS_ENABLE_SHADER_ERROR_COMPILATION_LOGGING
            glGetProgramiv (sp, GL_LINK_STATUS, &error);
    	    if (!error) {
    	        glGetProgramInfoLog (sp, SIZE, nullptr, errorBuffer);
		    	LOGERROR ("ERROR::SHADER::COMPUTE::LINKING_FAILED\n");
		    	ERROR ("MSG [%d]: \n%s\n", errorSize, errorBuffer);
    	    }
        #endif

    }

}

namespace SHADER {

	void Compile (
		OUT		GLuint& 		sp,	 // shader-program
		IN		const c8* const vsd, // vertex-shader-data
		IN		const c8* const fsd	 // fragment-shader-data
	) {

        #ifdef ATS_ENABLE_SHADER_ERROR_COMPILATION_LOGGING
		    const auto SIZE = 1024;
    	    c8 errorBuffer[SIZE];
		    s32 errorSize = 0;
		    GLint error;
        #endif

		// Vertex Shader
    	GLuint vs = glCreateShader (GL_VERTEX_SHADER);
    	glShaderSource (vs, 1, &vsd, nullptr);
    	glCompileShader (vs);

        #ifdef ATS_ENABLE_SHADER_ERROR_COMPILATION_LOGGING
    	    glGetShaderiv (vs, GL_COMPILE_STATUS, &error);
    	    if (!error) { // ERROR
    	        glGetShaderInfoLog (vs, SIZE, &errorSize, errorBuffer);
		    	LOGERROR ("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n");
                glDeleteShader (vs);
		    	ERROR ("MSG [%d]: \n%s\n", errorSize, errorBuffer); 
    	    }
        #endif

        // Fragment Shader
    	GLuint fs = glCreateShader (GL_FRAGMENT_SHADER);

        #ifdef ATS_GLSL_INJECTION

            //  NOTE
            // I don't have to create my own parser I just need to Inject the code.

            u32 fsdLength = 0; for (; fsd[fsdLength] != 0; ++fsdLength);

            memcpy (GLSL::ffsd, fsd, GLSL_INJECTION_VERSION_OFFSET);
            memcpy (GLSL::ffsd + GLSL_INJECTION_VERSION_OFFSET, GLSL_INJECTION_INCLUDE_ALL, GLSL_INJECTION_INCLUDE_ALL_LENGTH);
            memcpy (GLSL::ffsd + GLSL_INJECTION_VERSION_OFFSET + GLSL_INJECTION_INCLUDE_ALL_LENGTH, fsd + GLSL_INJECTION_VERSION_OFFSET, fsdLength - GLSL_INJECTION_VERSION_OFFSET);
            memset (GLSL::ffsd + GLSL_INJECTION_INCLUDE_ALL_LENGTH + fsdLength, 0, 1);

            glShaderSource (fs, 1, &GLSL::ffsd, nullptr);

        #else

    	    glShaderSource (fs, 1, &fsd, nullptr);

        #endif

    	glCompileShader (fs);

        #ifdef ATS_ENABLE_SHADER_ERROR_COMPILATION_LOGGING
    	    glGetShaderiv (fs, GL_COMPILE_STATUS, &error);
    	    if (!error) {
    	        glGetShaderInfoLog (fs, SIZE, &errorSize, errorBuffer);
		    	LOGERROR ("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n");
                glDeleteShader (fs);
		    	ERROR ("MSG [%d]: \n%s\n", errorSize, errorBuffer);
    	    }
        #endif

    	// --- finally link both shaders into one.
    	sp = glCreateProgram ();
    	glAttachShader (sp, vs);
    	glAttachShader (sp, fs);
    	glLinkProgram (sp);
        // ---

        // --- Deallocate handlers
    	glDeleteShader (vs);
    	glDeleteShader (fs);
        // ---

        #ifdef ATS_ENABLE_SHADER_ERROR_COMPILATION_LOGGING
    	    glGetProgramiv (sp, GL_LINK_STATUS, &error);
    	    if (!error) {
    	        glGetProgramInfoLog (sp, SIZE, nullptr, errorBuffer);
		    	LOGERROR ("ERROR::SHADER::PROGRAM::LINKING_FAILED\n");
		    	ERROR ("MSG [%d]: \n%s\n", errorSize, errorBuffer);
    	    }
        #endif

	}


	void ReadFile (
		IN		const c8* const& 	filepath,
		IN		c8*& 				buffer, 
		IN		const u16& 			bufferSize,
        IN		const c8* const& 	errorMsg
	) {
		FILE* file = fopen (filepath, "rb");

		if (file == nullptr) ERROR ("Failed to open '%s' as %s shader file!\n", filepath, errorMsg);

		auto readCode = fread (buffer, sizeof (c8), bufferSize, file);
		fclose (file);

        if (readCode > bufferSize) ERROR ("Failed to load all data of '%s' as %s shader file!\n", filepath, errorMsg);

		buffer[readCode] = '\0'; // Replace '0xcd' with '\0'
	}


	void DeleteShaders (
		IN 		const u8& 						shadersCount,
		IN 		const GLuint* const& 			shaders
	) {
		for (u32 i = 0; i < shadersCount; ++i) {
			glDeleteProgram (shaders[i]);
		}
	}


	void Use (
		IN 		const GLuint& 					shader
	) {
		glUseProgram (shader);
	}


	//  ABOUT
	// It sets all the uniforms. Not all used by a MESH but all defined uniforms
	//  even tho a mesh does not use them. TODO.
	//
	// void Set (
	// 	IN 		const u8& 						uniformsCount,
	// 	IN 		const UNIFORM::Uniform* const& 	uniforms,
	// 	IN 		const u16* const& 				uniformsList
	// ) {
	// 	for (u8 i = 0; i < uniformsCount; ++i) {
	// 		auto uniform = uniforms[i];
	// 		uniform.jump (i, uniform.data);
	// 	}
	// }

	void Set (
		IN 		const UNIFORM::Uniform* const& 	uniforms,
		IN 		const u16* const& 				uniformsList
	) {
		const auto& count = uniformsList[0];
		for (u8 i = 0; i < count; ++i) {
			auto uniformID = uniformsList[i + 1];
			auto uniform = uniforms[uniformID];
			uniform.jump (i, uniform.data);
		}
	}

}
