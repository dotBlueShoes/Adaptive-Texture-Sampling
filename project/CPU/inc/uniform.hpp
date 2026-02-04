// Made by Matthew Strumillo 2024.07.20
//
#pragma once
#include <glib/uniform.hpp>

namespace UNIFORM {

	// BUFFERS.
	r32 projection	[1 + 16];
	r32 transform	[1 + 16];
	r32 view		[1 + 16];
	u32 texture0    [2] { 0 };
    u32 texture1    [2] { 0 };
    u32 texture2    [2] { 0 };
	r32 color		[4] { 0 };


    #if ATS_MSAA_LEVEL == 0

        const u32 uniformsCount = 7;

	    // Each uniform references a buffer and a jump in jump-table.
	    Uniform uniforms [uniformsCount] {
	    	{ UNIFORM::JUMPS::SR4, &UNIFORM::color 		},
	    	{ UNIFORM::JUMPS::ST1, &UNIFORM::texture0 	},
            { UNIFORM::JUMPS::ST1, &UNIFORM::texture1 	},
            { UNIFORM::JUMPS::ST1, &UNIFORM::texture2 	},
	    	{ UNIFORM::JUMPS::M44, UNIFORM::projection 	},
	    	{ UNIFORM::JUMPS::M44, UNIFORM::view 		},
	    	{ UNIFORM::JUMPS::M44, UNIFORM::transform 	},
	    };

        // Identify what uniform is being used by what shader.
	    enum ID: u16 {
	    	COLOR 		= 0,
	    	TEXTURE0 	= 1,
            TEXTURE1 	= 2,
            TEXTURE2 	= 3,
	    	PROJECTION 	= 4,
	    	VIEW 		= 5,
	    	TRANSFORM 	= 6,
	    };

    #else

        const u32 uniformsCount = 7;

        // Each uniform references a buffer and a jump in jump-table.
	    Uniform uniforms [uniformsCount] {
	    	{ UNIFORM::JUMPS::SR4, &UNIFORM::color 		},
	    	{ UNIFORM::JUMPS::ST1, &UNIFORM::texture0 	},
            { UNIFORM::JUMPS::SM1, &UNIFORM::texture0 	},
            { UNIFORM::JUMPS::ST1, &UNIFORM::texture1 	},
	    	{ UNIFORM::JUMPS::M44, UNIFORM::projection 	},
	    	{ UNIFORM::JUMPS::M44, UNIFORM::view 		},
	    	{ UNIFORM::JUMPS::M44, UNIFORM::transform 	},
	    };

        // Identify what uniform is being used by what shader.
	    enum ID: u16 {
	    	COLOR 		        = 0,
	    	TEXTURE0 	        = 1,
            MULTISAMPLE_TEXTURE = 2,
            TEXTURE1 	        = 3,
	    	PROJECTION 	        = 4,
	    	VIEW 		        = 5,
	    	TRANSFORM 	        = 6,
	    };

    #endif

}