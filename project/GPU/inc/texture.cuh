#pragma once
//
#include "cuda/framework.cuh"

namespace TEXTURE {

    __global__ void ProcessSurface (
        /* IN  */ cudaSurfaceObject_t   surface, 
        /* IN  */ u32                   width, 
        /* IN  */ u32                   height
    ) {
        u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        u32 y = blockIdx.y * blockDim.y + threadIdx.y;

        // Only needed if the image would not be 32x.
        // if (x >= width || y >= height) return;

        // Why 4 ? -> CUDA arrays often store data as 4-channel formats (like uchar4) even if OpenGL texture was set as RGB only.
        uchar4 color = make_uchar4 (255, 0, 0, 255); // Example: write red.
        surf2Dwrite (color, surface, x * sizeof (uchar4), y);
    }

}


namespace TEXTURE {

    // TODO
    //  The following is a simple test. It does not operate on mipmaps
    //  or work with the loaded image in any shape or form.

    void Create (
        /* OUT */ GLuint&               glTexture,
		/* IN  */ const u16&            width, 
        /* IN  */ const u16&            height
    ) {
        cudaError_t errorCuda;

        cudaGraphicsResource_t cudaTexture;
        cudaSurfaceObject_t surface;
        cudaArray_t texture;
                        

        glBindTexture (GL_TEXTURE_2D, glTexture);

        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);


        // Hook OpenGL texture into CUDA.
        // 'cudaGraphicsRegisterFlagsSurfaceLoadStore' -> CUDA will write to the resource and 
        //   discard previous contents.
        errorCuda = cudaGraphicsGLRegisterImage (
            &cudaTexture, 
            glTexture, 
            GL_TEXTURE_2D, 
            cudaGraphicsRegisterFlagsSurfaceLoadStore
        ); CUDA_GET_ERROR (errorCuda, "cudaGraphicsGLRegisterImage-texture");


        // Temporarily gain access for CUDA to read/write it.
        errorCuda = cudaGraphicsMapResources (1, &cudaTexture, 0);
        CUDA_GET_ERROR (errorCuda, "cudaGraphicsMapResources-texture");


        // This gives us access to the actual image memory.
        errorCuda = cudaGraphicsSubResourceGetMappedArray (&texture, cudaTexture, 0, 0); 
        CUDA_GET_ERROR (errorCuda, "cudaGraphicsSubResourceGetMappedArray-texture");


        // Describe the use case.
        cudaResourceDesc descriptor = {};
        descriptor.resType = cudaResourceTypeArray;
        descriptor.res.array.array = texture;


        // Wraps the array in a HELPER so CUDA kernels can read/write it efficiently.
        errorCuda = cudaCreateSurfaceObject (&surface, &descriptor);
        CUDA_GET_ERROR (errorCuda, "cudaCreateSurfaceObject-texture");


        { // Process the texture on INPUT.
            //const dim3 threads (16, 16); // 256, 2
            //const dim3 blocks ((width  + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y); // 400, 2

            const dim3 threads (32, 32);
            const dim3 blocks  (width / 32, height / 32);

            ProcessSurface <<<blocks, threads>>> (surface, width, height);
            KERNEL_GET_ERROR ("ProcessSurface-texture");
        }
                        

        // Free HELPER and release the underlying CUDA internal handles and state.
        errorCuda = cudaDestroySurfaceObject (surface);
        CUDA_GET_ERROR (errorCuda, "cudaDestroySurfaceObject-texture");


        // Remove temporary access of CUDA to read/write to the texture.
        errorCuda = cudaGraphicsUnmapResources (1, &cudaTexture, 0);
        CUDA_GET_ERROR (errorCuda, "cudaGraphicsUnmapResources-texture");


        // Permanently detach the OpenGL texture from CUDA.
        errorCuda = cudaGraphicsUnregisterResource (cudaTexture);
        CUDA_GET_ERROR (errorCuda, "cudaGraphicsUnregisterResource-texture");

        glBindTexture (GL_TEXTURE_2D, 0);
    }

}
