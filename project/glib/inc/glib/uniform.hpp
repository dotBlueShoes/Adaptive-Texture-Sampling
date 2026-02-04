// Made by Matthew Strumillo 2024.03.08
//
#pragma once
#include "gl.hpp"

//  'About'
// - 'location' is not used ! // programmer has to know the sequence.

namespace GLM {

	struct VR1 {
		GLfloat x;
	};

	struct VR2 {
		GLfloat x, y;
	};

	struct VR3 {
		GLfloat x, y, z;
	};

	struct VR4 {
		GLfloat x, y, z, w;
	};

	struct VS1 {
		GLint x;
	};

	struct VS2 {
		GLint x, y;
	};

	struct VS3 {
		GLint x, y, z;
	};

	struct VS4 {
		GLint x, y, z, w;
	};

	struct VU1 {
		GLuint x;
	};

	struct VU2 {
		GLuint x, y;
	};

	struct VU3 {
		GLuint x, y, z;
	};

	struct VU4 {
		GLuint x, y, z, w;
	};

	struct M22 {
		GLfloat x1, x2;
		GLfloat y1, y2;
	};

	struct M33 {
		GLfloat x1, x2, x3;
		GLfloat y1, y2, y3;
		GLfloat z1, z2, z3;
	};

	struct M44 {
		GLfloat x1, x2, x3, x4;
		GLfloat y1, y2, y3, y4;
		GLfloat z1, z2, z3, z4;
		GLfloat w1, w2, w3, w4;
	};

	struct M23 {
		GLfloat x1, x2;
		GLfloat y1, y2;
		GLfloat z1, z2;
	};

	struct M32 {
		GLfloat x1, x2, x3;
		GLfloat y1, y2, y3;
	};

	struct M24 {
		GLfloat x1, x2;
		GLfloat y1, y2;
		GLfloat z1, z2;
		GLfloat w1, w2;
	};

	struct M42 {
		GLfloat x1, x2, x3, x4;
		GLfloat y1, y2, y3, y4;
	};

	struct M34 {
		GLfloat x1, x2, x3;
		GLfloat y1, y2, y3;
		GLfloat z1, z2, z3;
		GLfloat w1, w2, w3;
	};

	struct M43 {
		GLfloat x1, x2, x3, x4;
		GLfloat y1, y2, y3, y4;
		GLfloat z1, z2, z3, z4;
	};

}

namespace UNIFORM::TYPE {

	struct SR1 {
		GLfloat x;
	};

	struct SR2 {
		GLfloat x, y;
	};

	struct SR3 {
		GLfloat x, y, z;
	};

	struct SR4 {
		GLfloat x, y, z, w;
	};

	struct SS1 {
		GLint x;
	};

	struct SS2 {
		GLint x, y;
	};

	struct SS3 {
		GLint x, y, z;
	};

	struct SS4 {
		GLint x, y, z, w;
	};

	struct SU1 {
		GLuint x;
	};

	struct SU2 {
		GLuint x, y;
	};

	struct SU3 {
		GLuint x, y, z;
	};

	struct SU4 {
		GLuint x, y, z, w;
	};

}

namespace UNIFORM {

	// 
	// M -> Matrix
	// V -> Vector
	// S -> Single
	// R -> Real
	// S -> Signed
	// U -> Unsigned
	//
	enum ENUM: u8 {
		SR1 =  0 + 1,
		SR2 =  1 + 1,
		SR3 =  2 + 1,
		SR4 =  3 + 1,
		SS1 =  4 + 1,
		SS2 =  5 + 1,
		SS3 =  6 + 1,
		SS4 =  7 + 1,
		SU1 =  8 + 1,
		SU2 =  9 + 1,
		SU3 = 10 + 1,
		SU4 = 11 + 1,
		VR1 = 12 + 1,
		VR2 = 13 + 1,
		VR3 = 14 + 1,
		VR4 = 15 + 1,
		VS1 = 16 + 1,
		VS2 = 17 + 1,
		VS3 = 18 + 1,
		VS4 = 19 + 1,
		VU1 = 20 + 1,
		VU2 = 21 + 1,
		VU3 = 22 + 1,
		VU4 = 23 + 1,
		M22 = 34 + 1,
		M33 = 35 + 1,
		M44 = 36 + 1,
		M23 = 37 + 1,
		M32 = 38 + 1,
		M24 = 39 + 1,
		M42 = 40 + 1,
		M34 = 41 + 1,
		M43 = 42 + 1,
		NON = 0,
	};

}

namespace UNIFORM::JUMPS {

	// 'GL_INVALID_OPERATION' is generated if there is no current program object.
	// 'GL_INVALID_OPERATION' is generated if the size of the uniform variable declared in the shader does not match the size indicated by the glUniform command.
	// 'GL_INVALID_OPERATION' is generated if one of the signed or unsigned integer variants of this function is used to load a uniform variable of type float, vec2, vec3, vec4, or an array of these, or if one of the floating-point variants of this function is used to load a uniform variable of type int, ivec2, ivec3, ivec4, unsigned int, uvec2, uvec3, uvec4, or an array of these.
	// 'GL_INVALID_OPERATION' is generated if one of the signed integer variants of this function is used to load a uniform variable of type unsigned int, uvec2, uvec3, uvec4, or an array of these.
	// 'GL_INVALID_OPERATION' is generated if one of the unsigned integer variants of this function is used to load a uniform variable of type int, ivec2, ivec3, ivec4, or an array of these.
	// 'GL_INVALID_OPERATION' is generated if location is an invalid uniform location for the current program object and location is not equal to -1.
	// 'GL_INVALID_VALUE'	  is generated if count is less than 0.
	// 'GL_INVALID_OPERATION' is generated if count is greater than 1 and the indicated uniform variable is not an array variable.
	// 'GL_INVALID_OPERATION' is generated if a sampler is loaded using a command other than glUniform1i and glUniform1iv.

	using Jump = void(*) (const GLint&, void*);

	
	void ST1 (const GLint& location, void* data) {
		const auto su2 = *(UNIFORM::TYPE::SU2*)data;
		const auto textureLocation = su2.x;                 // Which 'active' texture connect to which 'texture' uniform.
		const auto texture = su2.y;                         // What is the 'active' texture.

        glActiveTexture (GL_TEXTURE0 + textureLocation);    // IMPORTANT. Ensure the texture uniforms are always first!
		glBindTexture (GL_TEXTURE_2D, texture);
		glUniform1i (location, textureLocation);
		GL::GetError ("glUniform1i-tex");
	}

    void SM1 (const GLint& location, void* data) {
		const auto su2 = *(UNIFORM::TYPE::SU2*)data;
		const auto textureLocation = su2.x;                 // Which 'active' texture connect to which 'texture' uniform.
		const auto texture = su2.y;                         // What is the 'active' texture.

        glActiveTexture (GL_TEXTURE0 + textureLocation);    // IMPORTANT. Ensure the texture uniforms are always first!
		glBindTexture (GL_TEXTURE_2D_MULTISAMPLE, texture);
		glUniform1i (location, textureLocation);
		GL::GetError ("glUniform1i-multex");
	}

	void SR1 (const GLint& location, void* data) {
		const auto sr1 = *(UNIFORM::TYPE::SR1*)data;
		glUniform1f (location, sr1.x);
		GL::GetError ("glUniform1f");
	}

	void SR2 (const GLint& location, void* data) {
		const auto sr2 = *(UNIFORM::TYPE::SR2*)data;
		glUniform2f (location, sr2.x, sr2.y);
		GL::GetError ("glUniform2f");
	}

	void SR3 (const GLint& location, void* data) {
		const auto sr3 = *(UNIFORM::TYPE::SR3*)data;
		glUniform3f (location, sr3.x, sr3.y, sr3.z);
		GL::GetError ("glUniform3f");
	}

	void SR4 (const GLint& location, void* data) {
		const auto sr4 = *(UNIFORM::TYPE::SR4*)data;
		glUniform4f (location, sr4.x, sr4.y, sr4.z, sr4.w);
		GL::GetError ("glUniform4f");
	}

	void SS1 (const GLint& location, void* data) {
		const auto ss1 = *(UNIFORM::TYPE::SS1*)data;
		glUniform1i (location, ss1.x);
		GL::GetError ("glUniform1i");
	}

	void SS2 (const GLint& location, void* data) {
		const auto ss2 = *(UNIFORM::TYPE::SS2*)data;
		glUniform2i (location, ss2.x, ss2.y);
		GL::GetError ("glUniform2i");
	}

	void SS3 (const GLint& location, void* data) {
		const auto ss3 = *(UNIFORM::TYPE::SS3*)data;
		glUniform3i (location, ss3.x, ss3.y, ss3.z);
		GL::GetError ("glUniform3i");
	}

	void SS4 (const GLint& location, void* data) {
		const auto ss4 = *(UNIFORM::TYPE::SS4*)data;
		glUniform4i (location, ss4.x, ss4.y, ss4.z, ss4.w);
		GL::GetError ("glUniform4i");
	}

	void SU1 (const GLint& location, void* data) {
		const auto su1 = *(UNIFORM::TYPE::SU1*)data;
		glUniform1ui (location, su1.x);
		GL::GetError ("glUniform1ui");
	}

	void SU2 (const GLint& location, void* data) {
		const auto su2 = *(UNIFORM::TYPE::SU2*)data;
		glUniform2ui (location, su2.x, su2.y);
		GL::GetError ("glUniform2ui");
	}

	void SU3 (const GLint& location, void* data) {
		const auto su3 = *(UNIFORM::TYPE::SU3*)data;
		glUniform3ui (location, su3.x, su3.y, su3.z);
		GL::GetError ("glUniform3ui");
	}

	void SU4 (const GLint& location, void* data) {
		const auto su4 = *(UNIFORM::TYPE::SU4*)data;
		glUniform4ui (location, su4.x, su4.y, su4.z, su4.w);
		GL::GetError ("glUniform4ui");
	}

	void VR1 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniform1fv (location, count, values);
		GL::GetError ("glUniform1fv");
	}

	void VR2 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniform2fv (location, count, values);
		GL::GetError ("glUniform2fv");
	}

	void VR3 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniform3fv (location, count, values);
		GL::GetError ("glUniform3fv");
	}

	void VR4 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniform4fv (location, count, values);
		GL::GetError ("glUniform4fv");
	}

	void VS1 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLint*)(data) + 1);
		glUniform1iv (location, count, values);
		GL::GetError ("glUniform1iv");
	}

	void VS2 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLint*)(data) + 1);
		glUniform2iv (location, count, values);
		GL::GetError ("glUniform2iv");
	}

	void VS3 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLint*)(data) + 1);
		glUniform3iv (location, count, values);
		GL::GetError ("glUniform3iv");
	}

	void VS4 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLint*)(data) + 1);
		glUniform4iv (location, count, values);
		GL::GetError ("glUniform4iv");
	}

	void VU1 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLuint*)(data) + 1);
		glUniform1uiv (location, count, values);
		GL::GetError ("glUniform1uiv");
	}

	void VU2 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLuint*)(data) + 1);
		glUniform2uiv (location, count, values);
		GL::GetError ("glUniform2uiv");
	}

	void VU3 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLuint*)(data) + 1);
		glUniform3uiv (location, count, values);
		GL::GetError ("glUniform3uiv");
	}

	void VU4 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLuint*)(data) + 1);
		glUniform4uiv (location, count, values);
		GL::GetError ("glUniform4uiv");
	}

	void M22 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniformMatrix2fv (location, count, false, values);
		GL::GetError ("glUniformMatrix2fv");
	}

	void M33 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniformMatrix3fv (location, count, false, values);
		GL::GetError ("glUniformMatrix3fv");
	}

	void M44 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniformMatrix4fv (location, count, false, values);
		GL::GetError ("glUniformMatrix4fv");
	}

	void M23 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniformMatrix2x3fv (location, count, false, values);
		GL::GetError ("glUniformMatrix2x3fv");
	}

	void M32 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniformMatrix3x2fv (location, count, false, values);
		GL::GetError ("glUniformMatrix3x2fv");
	}

	void M24 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniformMatrix2x4fv (location, count, false, values);
		GL::GetError ("glUniformMatrix2x4fv");
	}

	void M42(const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniformMatrix4x2fv (location, count, false, values);
		GL::GetError ("glUniformMatrix4x2fv");
	}

	void M34 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniformMatrix3x4fv (location, count, false, values);
		GL::GetError ("glUniformMatrix3x4fv");
	}

	void M43 (const GLint& location, void* data) {
		const auto count = ((GLint*)(data))[0];
		const auto values = ((GLfloat*)(data) + 1);
		glUniformMatrix4x3fv (location, count, false, values);
		GL::GetError ("glUniformMatrix4x3fv");
	}

}

namespace UNIFORM {

	struct Uniform {
		JUMPS::Jump jump;
		//GLint location;
		void* data;
	};

}
