// Made by Matthew Strumillo 2025.03.19
//
#pragma once
#include "gl.hpp"

//  ABOUT
// Element is an instance of a Point geometry which is a vertex.
// A 3D Vertex is defined using 3 coordinates.
//

namespace MESH::DDD {

	const u8 VERTEX = 3;
	const u8 COLOR 	= 3;
	const u8 UV 	= 2;

}

namespace MESH::DDD::CUBE {

	const u32 MODE = GL_TRIANGLES;
	const u32 ELEMENTS_COUNT = 36;
	const u32 POINTS_COUNT = 36;

	const u32 ELEMENTS [] {
		 0,  1,  2,
		 3,  4,  5,

		 6,  7,  8,
		 9, 10, 11,

		12, 13, 14,
		15, 16, 17,

		18, 19, 20,
		21, 22, 23,

		24, 25, 26,
		27, 28, 29,

		30, 31, 32,
		33, 34, 35,
	};

	const r32 VERTICES [] {
        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,

        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f,  0.5f,

        -0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,

         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,

        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
        -0.5f, -0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f,

        -0.5f,  0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f, -0.5f,
    };

	const r32 UVS [] {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,

        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,

        // ISSUE. flipped this side only.

        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,

        //
        //1.0f, 0.0f,
        //1.0f, 1.0f,
        //0.0f, 1.0f,
        //0.0f, 1.0f,
        //0.0f, 0.0f,
        //1.0f, 0.0f,
        //

        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f,

        0.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 1.0f,

        0.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 1.0f
    };

}

namespace MESH::DDD::HALFCUBE {

	//  ABOUT
	// Fastest way to render a cube.
	//

	const u32 MODE = GL_TRIANGLE_FAN;
	const u32 ELEMENTS_COUNT = 8;
	const u32 POINTS_COUNT = 8;

	const u32 ELEMENTS [] {
		0, 1, 2,
		3, 4, 5,
		6, 7
	};

	const r32 VERTICES [] {
		 0.5f,  0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
	   	-0.5f, -0.5f, -0.5f,
		-0.5f,  0.5f, -0.5f,
		-0.5f,  0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f, -0.5f, -0.5f,
   };

	const r32 UVS[] {
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
	};
}

namespace MESH::DDD::MIRRORCUBE {

	//  ABOUT
	// Fast and texture is being mirrored on every side.
	//

	const u32 MODE = GL_TRIANGLE_STRIP;
	const u32 ELEMENTS_COUNT = 14;
	const u32 POINTS_COUNT = 14;

	const u32 ELEMENTS [] {
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 10, 11,
		12, 13
	};

	const r32 VERTICES [] {
		 0.5f,  0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
	    -0.5f,  0.5f, -0.5f,
	   	-0.5f, -0.5f, -0.5f,
		-0.5f, -0.5f,  0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f,  0.5f, -0.5f,
		 0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f, -0.5f,
		-0.5f,  0.5f,  0.5f,
		-0.5f, -0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		 0.5f, -0.5f,  0.5f,
   };

	const r32 UVS[] {
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		0.0f, 1.0f,
	};
}

namespace MESH::DDD::SQUARE {

	const u32 MODE = GL_TRIANGLE_STRIP;
	const u32 ELEMENTS_COUNT = 4;
	const u32 POINTS_COUNT = 4;

	const u32 ELEMENTS [] {
		0, 1, 2,
		3
	};

	const r32 VERTICES [] {
		 0.5f,  0.5f, 0.0f,
		 0.5f, -0.5f, 0.0f,
	    -0.5f,  0.5f, 0.0f,
	   	-0.5f, -0.5f, 0.0f,
   };

	const r32 UVS[] {
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		0.0f, 1.0f,
	};

}

namespace MESH::DDD::FSQUARE {

	const u32 MODE = GL_TRIANGLE_STRIP;
	const u32 ELEMENTS_COUNT = 4;
	const u32 POINTS_COUNT = 4;

	const u32 ELEMENTS [] {
		0, 1, 2,
		3
	};

	const r32 VERTICES [] {
		 1.0f,  1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
	    -1.0f,  1.0f, 0.0f,
	   	-1.0f, -1.0f, 0.0f,
   };

	const r32 UVS[] {
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		0.0f, 1.0f,
	};
	
}
