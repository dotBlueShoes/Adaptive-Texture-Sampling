// Made by Matthew Strumillo 2024.02.28
//
#pragma once
#include "meshes_ddd.hpp"
#include "meshes_dd.hpp"

namespace MESH {

	const u8 R32 = sizeof (r32); 

}

namespace MESH::BUFFER {

	void CreateVertex (
		IN  	const GLuint& 		buffer, 
		IN  	const u32& 			location,
		IN  	const u32& 			dataSize,
		IN  	const r32* const& 	data
	) {
		glBindBuffer (GL_ARRAY_BUFFER, buffer);
		glBufferData (GL_ARRAY_BUFFER, dataSize, data, GL_STATIC_DRAW);
		GL::GetError ("glBufferData-vbo");

		glVertexAttribPointer (location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof (r32), nullptr);
		glEnableVertexAttribArray (location);
		GL::GetError ("glEnableVertexAttribArray-vbo");
	}

	void CreateColor (
		IN  	const GLuint& 		buffer, 
		IN  	const u32& 			location,
		IN  	const u32& 			dataSize,
		IN  	const r32* const& 	data
	) {
		glBindBuffer (GL_ARRAY_BUFFER, buffer);
		glBufferData (GL_ARRAY_BUFFER, dataSize, data, GL_STATIC_DRAW);
		GL::GetError ("glBufferData-clr");

		glVertexAttribPointer (location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof (r32), nullptr);
		glEnableVertexAttribArray (location);
		GL::GetError ("glEnableVertexAttribArray-clr");
	}

	void CreateUV (
		IN  	const GLuint& 		buffer, 
		IN  	const u32& 			location,
		IN  	const u32& 			dataSize,
		IN  	const r32* const& 	data
	) {
		glBindBuffer (GL_ARRAY_BUFFER, buffer);
		glBufferData (GL_ARRAY_BUFFER, dataSize, data, GL_STATIC_DRAW);
		GL::GetError ("glBufferData-uvs");

		glVertexAttribPointer (location, 2, GL_FLOAT, GL_FALSE, 2 * sizeof (r32), nullptr);
		glEnableVertexAttribArray (location);
		GL::GetError ("glEnableVertexAttribArray-uvs");
	}

	void CreateElement (
		IN  	const GLuint& 		buffer,
		IN  	const u32& 			dataSize,
		IN  	const u32* const& 	data
	) {
		glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, buffer);
		glBufferData (GL_ELEMENT_ARRAY_BUFFER, dataSize, data, GL_STATIC_DRAW);
		GL::GetError ("glBufferData-ebo");
	}

}


namespace MESH::VAO {

	//
	// Vertex, Element
	//
	void CreateVE ( 
		INOUT 	GLuint& 			VAO,
		IN  	const u32& 			pointsSize,
		IN  	const r32* const& 	vertices,
		IN  	const u32& 			elementsSize,
		IN  	const u32* const& 	elements
	) {
		const u32 LOCATION_0_VERTEX 	= 0;
		const u32 LOCATION_1_ELEMENT 	= 1; // ... does not have an actual location in glsl.

		const u8 buffersCount = 2;
		GLuint buffers[buffersCount];

		const auto vbo = buffers + LOCATION_0_VERTEX;
		const auto ebo = buffers + LOCATION_1_ELEMENT;

		glGenBuffers (buffersCount, buffers);
		glBindVertexArray (VAO);

		BUFFER::CreateVertex (*vbo, LOCATION_0_VERTEX, pointsSize * DDD::VERTEX * R32, vertices);
		BUFFER::CreateElement (*ebo, elementsSize * R32, elements);

		glBindBuffer (GL_ARRAY_BUFFER, 0);  // 'GL_ARRAY_BUFFER' are stored in VAO so we can safely unbind them.
		glBindVertexArray (0); 				// We don't have to do this but I will for extra security.
	}

	//
	// Vertex, Color, Element
	//
	void CreateVCE ( 
		INOUT 	GLuint& 			VAO,
		IN  	const u32& 			pointsSize,
		IN  	const r32* const& 	vertices,
		IN  	const r32* const& 	colors,
		IN  	const u32& 			elementsSize,
		IN  	const u32* const& 	elements
	) {
		const u64 LOCATION_0_VERTEX 	= 0;
		const u64 LOCATION_1_COLOR 		= 1;
		const u64 LOCATION_2_ELEMENT 	= 2; // ... does not have an actual location in glsl.

		const u8 buffersCount = 3;
		GLuint buffers[buffersCount];

		auto vbo = buffers + LOCATION_0_VERTEX;
		auto clr = buffers + LOCATION_1_COLOR;
		auto ebo = buffers + LOCATION_2_ELEMENT;

		glGenBuffers (buffersCount, buffers);
		glBindVertexArray (VAO);

		BUFFER::CreateVertex (*vbo, LOCATION_0_VERTEX, pointsSize * DDD::VERTEX * R32, vertices);
		BUFFER::CreateColor (*clr, LOCATION_1_COLOR, pointsSize * DDD::COLOR * R32, colors);
		BUFFER::CreateElement (*ebo, elementsSize * R32, elements);

		glBindBuffer (GL_ARRAY_BUFFER, 0);  // 'GL_ARRAY_BUFFER' are stored in VAO so we can safely unbind them.
		glBindVertexArray (0); 				// We don't have to do this but I will for extra security.
	}


	

	//
	// Vertex, Uvs, Element
	//
	const u8 VUE_BUFFERS_COUNT = 3;
	//
	void CreateVUE ( 
		INOUT 	GLuint& 			VAO,
		INOUT	GLuint* const& 		buffers,
		IN  	const u32& 			pointsSize,
		IN  	const r32* const& 	vertices,
		IN  	const r32* const& 	uvs,
		IN  	const u32& 			elementsSize,
		IN  	const u32* const& 	elements
	) {
		const u64 LOCATION_0_VERTEX 	= 0;
		const u64 LOCATION_1_UVS 		= 1;
		const u64 LOCATION_2_ELEMENT 	= 2; // ... does not have an actual location in glsl.

		auto vbo = buffers + LOCATION_0_VERTEX;
		auto uvb = buffers + LOCATION_1_UVS;
		auto ebo = buffers + LOCATION_2_ELEMENT;

		glBindVertexArray (VAO);

		BUFFER::CreateVertex (*vbo, LOCATION_0_VERTEX, pointsSize * DDD::VERTEX * R32, vertices);
		BUFFER::CreateUV (*uvb, LOCATION_1_UVS, pointsSize * DDD::UV * R32, uvs);
		BUFFER::CreateElement (*ebo, elementsSize * R32, elements);

		glBindBuffer (GL_ARRAY_BUFFER, 0);  // 'GL_ARRAY_BUFFER' are stored in VAO so we can safely unbind them.
		glBindVertexArray (0); 				// We don't have to do this but I will for extra security.
	}


    //
	// Vertex, Uvs, Uvs, Element
	//
	const u8 VUUE_BUFFERS_COUNT = 4;
	//
	void CreateVUUE ( 
		INOUT 	GLuint& 			VAO,
		INOUT	GLuint* const& 		buffers,
		IN  	const u32& 			pointsSize,
		IN  	const r32* const& 	vertices,
		IN  	const r32* const& 	uvs,
        IN  	const r32* const& 	uvsExtra,
		IN  	const u32& 			elementsSize,
		IN  	const u32* const& 	elements
	) {
		const u64 LOCATION_0_VERTEX 	= 0;
		const u64 LOCATION_1_UVS 		= 1;
        const u64 LOCATION_2_UVS 		= 2;
		const u64 LOCATION_2_ELEMENT 	= 3; // ... does not have an actual location in glsl.

		auto vbo = buffers + LOCATION_0_VERTEX;
		auto uvb = buffers + LOCATION_1_UVS;
        auto uv1 = buffers + LOCATION_2_UVS;
		auto ebo = buffers + LOCATION_2_ELEMENT;

		glBindVertexArray (VAO);

		BUFFER::CreateVertex (*vbo, LOCATION_0_VERTEX, pointsSize * DDD::VERTEX * R32, vertices);
		BUFFER::CreateUV (*uvb, LOCATION_1_UVS, pointsSize * DDD::UV * R32, uvs);
        BUFFER::CreateUV (*uv1, LOCATION_2_UVS, pointsSize * DDD::UV * R32, uvsExtra);
		BUFFER::CreateElement (*ebo, elementsSize * R32, elements);

		glBindBuffer (GL_ARRAY_BUFFER, 0);  // 'GL_ARRAY_BUFFER' are stored in VAO so we can safely unbind them.
		glBindVertexArray (0); 				// We don't have to do this but I will for extra security.
	}

}

// NAH
//#define DDD_SQUARE_BUFFERS 3
//#define DDD_SQUARE_CREATE(vao, buffers) MESH::VAO::CreateVUE (vao, buffers, MESH::DDD::SQUARE::POINTS_COUNT, MESH::DDD::SQUARE::VERTICES, MESH::DDD::SQUARE::UVS, MESH::DDD::SQUARE::ELEMENTS_COUNT, MESH::DDD::SQUARE::ELEMENTS);
//#define DDD_SQUARE_DRAW() glDrawElements (GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

// A mesh is
// - vao
// - buffers offset
// - create method
// - draw method

// TODO
// 1. A system for defining an existing and loaded meshes ()
// 2. A system for drawing an existing and loaded meshes (draw method + type + count)

//namespace MESH::VAO {
//
//	enum Type : u8 {
//		ERROR 		= 0,
//		CUSTOM_VE  	= 1,
//		CUSTOM_VCE 	= 2,
//		CUSTOM_VUE 	= 3,
//
//		SQUARE_VCE	= 4,
//		SQUARE_VUE	= 5,
//
//		CUBE_VCE	= 6,
//		CUBE_VUE	= 7,
//	};
//
//	const u32 BUFFER_COUNTS[4] { };
//	const u32 SHADER_CREATE_FUNCTIONS[4] { };
//	const u32 CREATE_FUNCTIONS[4] { };
//
//}


namespace MESH {

	using Draw = void (*) (const u32& elementsCount, const u8& drawMode);

	void DrawElements (const u32& elementsCount, const u8& drawMode)  {
		glDrawElements (drawMode, elementsCount, GL_UNSIGNED_INT, 0);
		GL::GetError ("glDrawElements");
		//LOGINFO ("call!\n");
	}

	Draw draws[] { 
		DrawElements 
	};

	struct Mesh {
		u16 vaoIndex;		// VAOS array index.
		u8 drawIndex;		// DRAWS array index.
		u32 elementsCount;	// 
		u8 drawMode;		// 
	};

}
