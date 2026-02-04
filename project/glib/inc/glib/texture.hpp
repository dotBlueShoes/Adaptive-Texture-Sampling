// Made by Matthew Strumillo 2024.03.08
//
#pragma once
#include "gl.hpp"
#include "image.hpp"

// TODO
// - Asserts for mipmapped and without for GL filtering values to ensure they match.

namespace TEXTURE {

    void CreateBase (
        /* OUT */ GLuint&               texture,
		/* IN  */ const u16&            iChannelsType,
        /* IN  */ const u16&            oChannelsType,
		/* IN  */ const u16&            width,
        /* IN  */ const u16&            height,
        /* IN  */ const u8* const&      imageData,
        /* IN  */ const GLint&          minFiltering,
        /* IN  */ const GLint&          magFiltering
    ) {

        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);	
		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFiltering);
		glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFiltering);
        GL::GetError ("glTexParameter");

		glTexImage2D (
			GL_TEXTURE_2D, 
			0, iChannelsType, 
			width, height, 
			0, oChannelsType, 
			GL_UNSIGNED_BYTE, 
			imageData
		);
    }


    // TODO. TEST it.
    void CreateMipmapped (
		/* OUT */ GLuint&               texture,
		/* IN  */ const IMAGE::Head&    imageHeader,
        /* IN  */ const u8* const&      imageData,
        /* IN  */ const GLint&          minFiltering,
        /* IN  */ const GLint&          magFiltering,
        /* IN  */ const u8&             mipmapsCount,
        /* IN  */ u8*                   mipmapData
	) {
		glBindTexture (GL_TEXTURE_2D, texture);
		GL::GetError ("glBindTexture");

        // Antisotropic filtering only works with mipmaps on.
        #if ATS_ANISOTROPY_LEVEL > 0
            glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, ATS_ANISOTROPY_LEVEL);
        #endif
		
		switch (imageHeader.channels) {

			case 3: {

                CreateBase (texture, GL_RGB8, GL_RGB, imageHeader.width, imageHeader.height, imageData, minFiltering, magFiltering);
				GL::GetError ("glTexImage2D-RGB");

                for (u8 iMipmap = 1; iMipmap < mipmapsCount + 1; ++iMipmap) {
                    
                    const auto& mipmapHeight = imageHeader.height / (2 << iMipmap);
                    const auto& mipmapWidth = imageHeader.width / (2 << iMipmap);

                    glTexImage2D (
                        GL_TEXTURE_2D, iMipmap, GL_RGB8, 
                        mipmapWidth, mipmapHeight, 
                        0, GL_RGB, GL_UNSIGNED_BYTE, 
                        mipmapData
                    );
                    GL_GET_ERROR ("glTexImage2D-mipmap-%d", iMipmap);

                    mipmapData += (mipmapHeight * mipmapWidth * 3);
                }

			} break;

			case 4: {

                CreateBase (texture, GL_RGBA8, GL_RGBA, imageHeader.width, imageHeader.height, imageData, minFiltering, magFiltering);
				GL::GetError ("glTexImage2D-RGBA");

			} break;

			default: {
				ERROR ("Unknown texture channels durion texture creation!\n");
			} 

		}
			
    	glGenerateMipmap (GL_TEXTURE_2D);
		GL::GetError ("glGenerateMipmap");

		glBindTexture (GL_TEXTURE_2D, 0); // We don't have to do this but I will for extra security.
	}


	void CreateMipmapped (
		/* OUT */ GLuint&               texture,
		/* IN  */ const IMAGE::Head&    imageHeader,
        /* IN  */ const u8* const&      imageData,
        /* IN  */ const GLint&          minFiltering,
        /* IN  */ const GLint&          magFiltering
	) {
		glBindTexture (GL_TEXTURE_2D, texture);
		GL::GetError ("glBindTexture");

        // Antisotropic filtering only works with mipmaps on.
        #if ATS_ANISOTROPY_LEVEL > 0
            glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, ATS_ANISOTROPY_LEVEL);
        #endif
		
		switch (imageHeader.channels) {

			case 3: {

                CreateBase (texture, GL_RGB8, GL_RGB, imageHeader.width, imageHeader.height, imageData, minFiltering, magFiltering);
				GL::GetError ("glTexImage2D-RGB");

			} break;

			case 4: {

                CreateBase (texture, GL_RGBA8, GL_RGBA, imageHeader.width, imageHeader.height, imageData, minFiltering, magFiltering);
				GL::GetError ("glTexImage2D-RGBA");

			} break;

			default: {
				ERROR ("Unknown texture channels durion texture creation!\n");
			} 

		}
			
    	glGenerateMipmap (GL_TEXTURE_2D);
		GL::GetError ("glGenerateMipmap");

		glBindTexture (GL_TEXTURE_2D, 0); // We don't have to do this but I will for extra security.
	}


    void Create (
		/* OUT */ GLuint&               texture,
		/* IN  */ const IMAGE::Head&    imageHeader,
        /* IN  */ const u8* const&      imageData,
        /* IN  */ const GLint&          minFiltering,
        /* IN  */ const GLint&          magFiltering
	) {
		glBindTexture (GL_TEXTURE_2D, texture);
		GL::GetError ("glBindTexture");
		
		switch (imageHeader.channels) {

			case 3: {

                CreateBase (texture, GL_RGB8, GL_RGB, imageHeader.width, imageHeader.height, imageData, minFiltering, magFiltering);
				GL::GetError ("glTexImage2D-RGB");

			} break;

			case 4: {

                CreateBase (texture, GL_RGBA8, GL_RGBA, imageHeader.width, imageHeader.height, imageData, minFiltering, magFiltering);
				GL::GetError ("glTexImage2D-RGBA");

			} break;

			default: {
				ERROR ("Unknown texture channels durion texture creation!\n");
			} 

		}

		glBindTexture (GL_TEXTURE_2D, 0); // We don't have to do this but I will for extra security.
	}

}
