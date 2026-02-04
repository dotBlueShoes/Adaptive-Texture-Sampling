// Made by Matthew Strumillo 2024.07.20
//
#pragma once
//
#include <stb_image.h>
#include <stb_image_write.h>
//
#include <blue/error.hpp>

namespace IMAGE {


    struct Head {
		u8 channels;
		u16 height;
		u16 width;
	};


	void Load (
		/* OUT */ Head&             header,
        /* OUT */ u8*&              imageData,
		/* IN  */ const c8* const&  filepath
	) {
		s32 channels;
		s32 width;
		s32 height;

		imageData = stbi_load (filepath, &width, &height, &channels, 0);
		if (imageData == nullptr) ERROR ("Incorrect image filepath!\n");

		{ // RET - Variables in proper type sizes.
			header.channels = channels; // <- Replace with an enum already?
			header.height = height;
			header.width = width;
		}

		INCALLOCCO ();
		
	}


	void SaveAsPNG (
		/* IN */ const Head&       header,
        /* IN */ const u8* const&  imageData,
		/* IN  */ const c8* const& filepath
	) {
		const auto strideBytes = header.width * header.channels;
		stbi_write_png (filepath, header.width, header.height, header.channels, imageData, strideBytes);
	}

	void Free (
		/* IN  */ void* handleData
	) {
		stbi_image_free (handleData);
		DECALLOCCO ();
	}

}
