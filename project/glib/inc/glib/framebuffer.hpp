#pragma once
#include "texture.hpp"
#include "gl.hpp"

namespace FRAMEBUFFER {

    //  ABOUT 
    // Check from what positions the subsamples are being taken.
    // -> It depends on GPU but it should be possible to tell the api how.
    //
    void PrintMsaaSampleCoords () {
        #if ATS_MSAA_LEVEL > 0
            GLfloat coord[2];

            for (u8 i = 0; i < ATS_MSAA_LEVEL; ++i) {
                glGetMultisamplefv (GL_SAMPLE_POSITION, i, coord);
                GL::GetError ("msaa-glGetMultisamplefv");
                LOGINFO ("i: x %f, y %f\n", coord[0], coord[1]);
            }
        #endif 
    }

}

namespace FRAMEBUFFER::TEXTURE {
    void Create (
        /* OUT */ GLuint&       texture,
        /* IN  */ const u16&    iChannelsType,
        /* IN  */ const u16&    oChannelsType,
        /* IN  */ const u16&    width,
        /* IN  */ const u16&    height,
        /* IN  */ const GLint&  filtering,
        /* IN  */ const GLuint& oType
    ) {
        glBindTexture (GL_TEXTURE_2D, texture);
        GL::GetError ("framebuffer-glBindTexture");

        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filtering);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filtering); 
        GL::GetError ("framebuffer-glTexParameter");

        glTexImage2D (
            GL_TEXTURE_2D, 
            0, iChannelsType, 
            width, height, 
            0, oChannelsType, 
            oType, 
            NULL
        );
        GL::GetError ("framebuffer-glTexImage2D");

        glBindTexture (GL_TEXTURE_2D, 0);
    }
}

namespace FRAMEBUFFER::MSAA::TEXTURE {

    void Create (
        /* OUT */ GLuint&       texture,
        /* IN  */ const u16&    iChannelsType,
        /* IN  */ const u16&    oChannelsType,
        /* IN  */ const u16&    width,
        /* IN  */ const u16&    height
    ) {
        glBindTexture (GL_TEXTURE_2D_MULTISAMPLE, texture);
        GL::GetError ("framebuffer-msaa-glBindTexture");

        glTexImage2DMultisample (
            GL_TEXTURE_2D_MULTISAMPLE, 
            ATS_MSAA_LEVEL, 
            iChannelsType, 
            width, 
            height, 
            GL_TRUE
        );

        GL::GetError ("framebuffer-msaa-glTexImage2DMultisample");

        glBindTexture (GL_TEXTURE_2D_MULTISAMPLE, 0);
    }
}


namespace FRAMEBUFFER::RENDERBUFFEROBJECT {

    void Create (
        /* OUT */ GLuint&       renderBufferObject,
        /* IN  */ const u16&    format,
        /* IN  */ const u16&    width,
        /* IN  */ const u16&    height
    ) {
        glBindRenderbuffer (GL_RENDERBUFFER, renderBufferObject); 
        GL::GetError ("framebuffer-glBindRenderbuffer");

        glRenderbufferStorage (
            GL_RENDERBUFFER, format, 
            width, 
            height
        );

        GL::GetError ("framebuffer-glRenderbufferStorage");

        glBindRenderbuffer (GL_RENDERBUFFER, 0);
    }

}


namespace FRAMEBUFFER::MSAA::RENDERBUFFEROBJECT {

    void Create (
        /* OUT */ GLuint&       renderBufferObject,
        /* IN  */ const u16&    format,
        /* IN  */ const u16&    width,
        /* IN  */ const u16&    height
    ) {
        glBindRenderbuffer (GL_RENDERBUFFER, renderBufferObject); 
        GL::GetError ("framebuffer-glBindRenderbuffer");

        glRenderbufferStorageMultisample (
            GL_RENDERBUFFER, ATS_MSAA_LEVEL, format, 
            width, 
            height
        );

        GL::GetError ("framebuffer-glRenderbufferStorage");

        glBindRenderbuffer (GL_RENDERBUFFER, 0);
    }

}


namespace FRAMEBUFFER::DEPTHFRAMEBUFFER {

    void Create (
        /* OUT */ GLuint&       framebuffer,
        /* IN  */ const GLuint& texture,
        /* IN  */ const u16&    textureAttachment,
        /* IN  */ const GLuint& depthTexture,
        /* IN  */ const u16&    depthTextureAttachment
    ) {
        glBindFramebuffer (GL_FRAMEBUFFER, framebuffer); 
        GL::GetError ("framebuffer-glBindFramebuffer");

        glFramebufferTexture2D (
            GL_FRAMEBUFFER, textureAttachment, 
            GL_TEXTURE_2D, texture, 0
        );

        GL::GetError ("framebuffer-glFramebufferTexture2D");

        glFramebufferTexture2D (
            GL_FRAMEBUFFER, depthTextureAttachment, 
            GL_TEXTURE_2D, depthTexture, 0
        );

        GL::GetError ("framebuffer-glFramebufferTexture2D-depth");

        if (glCheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            ERROR ("Could not create a valid framebuffer!\n");
        }

        glBindFramebuffer (GL_FRAMEBUFFER, 0); 
    }

    void Create (
        /* OUT */ GLuint&       framebuffer,
        /* IN  */ const GLuint& texture0,
        /* IN  */ const u16&    textureAttachment0,
        /* IN  */ const GLuint& texture1,
        /* IN  */ const u16&    textureAttachment1,
        /* IN  */ const GLuint& depthTexture,
        /* IN  */ const u16&    depthTextureAttachment
    ) {
        glBindFramebuffer (GL_FRAMEBUFFER, framebuffer); 
        GL::GetError ("framebuffer-glBindFramebuffer");

        glFramebufferTexture2D (
            GL_FRAMEBUFFER, textureAttachment0, 
            GL_TEXTURE_2D, texture0, 0
        );

        GL::GetError ("framebuffer-glFramebufferTexture2D-0");

        glFramebufferTexture2D (
            GL_FRAMEBUFFER, textureAttachment1, 
            GL_TEXTURE_2D, texture1, 0
        );

        GL::GetError ("framebuffer-glFramebufferTexture2D-1");

        GLenum drawBuffers[2] { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers (2, drawBuffers);

        glFramebufferTexture2D (
            GL_FRAMEBUFFER, depthTextureAttachment, 
            GL_TEXTURE_2D, depthTexture, 0
        );

        GL::GetError ("framebuffer-glFramebufferTexture2D-depth");

        if (glCheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            ERROR ("Could not create a valid framebuffer!\n");
        }

        glBindFramebuffer (GL_FRAMEBUFFER, 0); 
    }

}

namespace FRAMEBUFFER::DEPTHFRAMEBUFFER::MSAA {

    void Create (
        /* OUT */ GLuint&       framebuffer,
        /* IN  */ const GLuint& texture,
        /* IN  */ const u16&    textureAttachment,
        /* IN  */ const GLuint& depthTexture,
        /* IN  */ const u16&    depthTextureAttachment
    ) {
        glBindFramebuffer (GL_FRAMEBUFFER, framebuffer); 
        GL::GetError ("framebuffer-msaa-glBindFramebuffer");

        glFramebufferTexture2D (
            GL_FRAMEBUFFER, textureAttachment, 
            GL_TEXTURE_2D_MULTISAMPLE, texture, 0
        );

        GL::GetError ("framebuffer-msaa-glFramebufferTexture2D");

        glFramebufferTexture2D (
            GL_FRAMEBUFFER, depthTextureAttachment, 
            GL_TEXTURE_2D, depthTexture, 0
        );

        GL::GetError ("framebuffer-glFramebufferTexture2D-depth");

        if (glCheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            ERROR ("Could not create a valid msaa-framebuffer!\n");
        }

        DEBUG (DEBUG_FLAG_LOGGING) PrintMsaaSampleCoords ();

        glBindFramebuffer (GL_FRAMEBUFFER, 0); 
    }

}

namespace FRAMEBUFFER {

    void Create (
        /* OUT */ GLuint&       framebuffer,
        /* IN  */ const GLuint& texture,
        /* IN  */ const u16&    textureAttachment,
        /* IN  */ const GLuint& renderBufferObject,
        /* IN  */ const u16&    renderBufferObjectAttachment
    ) {
        glBindFramebuffer (GL_FRAMEBUFFER, framebuffer); 
        GL::GetError ("framebuffer-glBindFramebuffer");

        glFramebufferTexture2D (
            GL_FRAMEBUFFER, textureAttachment, 
            GL_TEXTURE_2D, texture, 0
        );

        GL::GetError ("framebuffer-glFramebufferTexture2D");

        glFramebufferRenderbuffer (
            GL_FRAMEBUFFER, renderBufferObjectAttachment, 
            GL_RENDERBUFFER, renderBufferObject
        );

        GL::GetError ("framebuffer-glFramebufferRenderbuffer");

        if (glCheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            ERROR ("Could not create a valid framebuffer!\n");
        }

        glBindFramebuffer (GL_FRAMEBUFFER, 0); 
    }

    void Create (
        /* OUT */ GLuint&       framebuffer,
        /* IN  */ const GLuint& texture0,
        /* IN  */ const u16&    textureAttachment0,
        /* IN  */ const GLuint& texture1,
        /* IN  */ const u16&    textureAttachment1,
        /* IN  */ const GLuint& renderBufferObject,
        /* IN  */ const u16&    renderBufferObjectAttachment
    ) {
        glBindFramebuffer (GL_FRAMEBUFFER, framebuffer); 
        GL::GetError ("framebuffer-glBindFramebuffer");

        glFramebufferTexture2D (
            GL_FRAMEBUFFER, textureAttachment0, 
            GL_TEXTURE_2D, texture0, 0
        );

        GL::GetError ("framebuffer-glFramebufferTexture2D-0");

        glFramebufferTexture2D (
            GL_FRAMEBUFFER, textureAttachment1, 
            GL_TEXTURE_2D, texture1, 0
        );

        GL::GetError ("framebuffer-glFramebufferTexture2D-1");

        GLenum drawBuffers[2] { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers (2, drawBuffers);

        glFramebufferRenderbuffer (
            GL_FRAMEBUFFER, renderBufferObjectAttachment, 
            GL_RENDERBUFFER, renderBufferObject
        );

        GL::GetError ("framebuffer-glFramebufferRenderbuffer");

        if (glCheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            ERROR ("Could not create a valid framebuffer!\n");
        }

        glBindFramebuffer (GL_FRAMEBUFFER, 0); 
    }

}

namespace FRAMEBUFFER::MSAA {

    void Create (
        /* OUT */ GLuint&       framebuffer,
        /* IN  */ const GLuint& texture,
        /* IN  */ const u16&    textureAttachment,
        /* IN  */ const GLuint& renderBufferObject,
        /* IN  */ const u16&    renderBufferObjectAttachment
    ) {
        glBindFramebuffer (GL_FRAMEBUFFER, framebuffer); 
        GL::GetError ("framebuffer-msaa-glBindFramebuffer");

        glFramebufferTexture2D (
            GL_FRAMEBUFFER, textureAttachment, 
            GL_TEXTURE_2D_MULTISAMPLE, texture, 0
        );

        GL::GetError ("framebuffer-msaa-glFramebufferTexture2D");

        glFramebufferRenderbuffer (
            GL_FRAMEBUFFER, renderBufferObjectAttachment, 
            GL_RENDERBUFFER, renderBufferObject
        );

        GL::GetError ("framebuffer-msaa-glFramebufferRenderbuffer");

        if (glCheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            ERROR ("Could not create a valid msaa-framebuffer!\n");
        }

        DEBUG (DEBUG_FLAG_LOGGING) PrintMsaaSampleCoords ();

        glBindFramebuffer (GL_FRAMEBUFFER, 0); 
    }

}
