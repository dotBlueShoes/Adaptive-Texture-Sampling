#pragma once
#include "framework.cuh"

// TODO - OPTIMALIZATIONS
// 1. Reuse variables. patch[4][4] and then col[4] then result is redundant and unneeded see if other things.
// 2. __shared__ memory vs global memory eg. cudaTextureObject_t
// 3. Heavy math / redundant heavy math out of the loop.
// 4. Keep the format true. Ensure switches from r32 to u8 are neceserry if used and properly min-maxed if needed.
//
// namespace EDGE {
// 
//     // A method. saving detail. - Smooths out an image
//     // B method. overwriting a detail. - Creates a clear new pixel
// 
// }


#define GRAYSCALE_A(r, g, b) \
    (0.299f * r) + (0.587f * g) + (0.114f * b);

#define GRAYSCALE_B(r, g, b) \
    (0.21f * r) + (0.72f * g) + (0.07f * b);

#define GRAYSCALE_C(r, g, b) \
    (r + g + b) / 3;

#define GRAYSCALE(r, g, b) \
    GRAYSCALE_A (r, g, b)


namespace LINEDRAW {

    //  NOTE (Why this is wrong.)
    // The following only works for like x2 scalling. thats wrong it should work for every scale-up operation
    // What should be happening is I should chceck whats the scaling of the selective row (an integer value).
    // Based on that information decide how wide the line should be. Then when drawing the line. Not only
    // draw a line but also fill the empty space properly. 


    // kernel that draws a dual-line from point a to point b with color 1 and color 2 down example
    //
    __global__ void DLD (
        /* IN  */ u16                   psx,
        /* IN  */ u16                   psy,
        /* IN  */ u16                   pex,
        /* IN  */ u16                   pey,
        /* OUT */ u8*                   output,
        /* IN  */ uchar3                colorA,
        /* IN  */ uchar3                colorB
    ) {
        //const u32 ls = (640 * 4 * psx) + (4 * psy);
        //const u32 le = (640 * 4 * pex) + (4 * pey);
        //
        //// Going down always. No absolute needed.
        ////
        //const u16 lengthy = pey - psy;
        //const u16 lengthx = pex - psx;
            //
        //output[ls + 0] = (u8) colorA.x;
        //output[ls + 1] = (u8) colorA.y;
        //output[ls + 2] = (u8) colorA.z;
        //output[ls + 3] = 255;
        //
        //output[le + 0] = (u8) colorB.x;
        //output[le + 1] = (u8) colorB.y;
        //output[le + 2] = (u8) colorB.z;
        //output[le + 3] = 255;

        const u16 length = 10;
        
        
        // NOPE. AFTER UPSCALING
        //const r32 scale = 2.0f;
        // // start point is in texture coords before upscaling
        // // we need to convert it into new texture coords (down) algorithm position
        // u32 x = (u32) (psx * scale);

        
        u16 fhlength = (length / 2); // first half length
        // u16 hhlength = (length / 4); // half half length

        //u16 shlength = fhlength + (length % 2);

        r32 sar = (r32) (colorA.x) / (length); // step-a-red
        r32 sag = (r32) (colorA.y) / (length); // step-a-gre
        r32 sab = (r32) (colorA.z) / (length); // step-a-blu
        r32 sbr = (r32) (colorB.x) / (length); // step-b-red
        r32 sbg = (r32) (colorB.y) / (length); // step-b-gre
        r32 sbb = (r32) (colorB.z) / (length); // step-b-blu

        { // main trail

            // This looks bad
            // const u16 trail1e = fhlength + hhlength; // trail 1 end point
            // const u16 trail2s = fhlength - hhlength; // trail 2 start point

            const u16 trail1e = fhlength;
            const u16 trail2s = fhlength;

            for (u16 i = 0; i < trail1e; ++i) {
                u32 ps = ((psy + i) * 640 * 4) + ((psx + 1) * 4); 
                u16 ri = trail1e - i; // reverse i

                r32 r = (sbr * (ri + fhlength)) + (sar * (i + 1));
                r32 g = (sbg * (ri + fhlength)) + (sag * (i + 1));
                r32 b = (sbb * (ri + fhlength)) + (sab * (i + 1));

                output[ps + 0] = (u8) r;
                output[ps + 1] = (u8) g;
                output[ps + 2] = (u8) b;
                output[ps + 3] = 255;
            }

            for (u16 i = trail2s; i < length; ++i) {
                u32 ps = ((psy + i) * 640 * 4) + ((psx + 2) * 4); 
                u16 ri = length - i; // reverse i

                r32 r = (sbr * ri) + (sar * i);
                r32 g = (sbg * ri) + (sag * i);
                r32 b = (sbb * ri) + (sab * i);

                output[ps + 0] = (u8) r;
                output[ps + 1] = (u8) g;
                output[ps + 2] = (u8) b;
                output[ps + 3] = 255;
            }

        }

        //{
        //
        //    for (u16 i = 0; i < fhlength; ++i) {
        //        u32 ps = ((psy + i) * 640 * 4) + ((psx + 2) * 4);
        //        u16 ri = fhlength - i; // reverse i
        //
        //        u8 offset = 2;
        //        r32 r = (sbr * (ri + fhlength + offset)) + (sar * i);
        //        r32 g = (sbg * (ri + fhlength + offset)) + (sag * i);
        //        r32 b = (sbb * (ri + fhlength + offset)) + (sab * i);
        //
        //        output[ps + 0] = (u8) r;
        //        output[ps + 1] = (u8) g;
        //        output[ps + 2] = (u8) b;
        //        output[ps + 3] = 255;
        //    }
        //
        //    for (u16 i = fhlength; i < length; ++i) {
        //        u32 ps = ((psy + i) * 640 * 4) + ((psx + 1) * 4); 
        //
        //        output[ps + 0] = (u8) colorA.x;
        //        output[ps + 1] = (u8) colorA.y;
        //        output[ps + 2] = (u8) colorA.z;
        //        output[ps + 3] = 255;
        //    }
        //    
        //}


    }

}


namespace EDGE::TEXTURE::FILTERING {

    __device__ const s8 kernels [8][9] {

        { // detail A
            -1, -1, -1,
            -1,  8, -1,
            -1, -1, -1,
        },
        { // detail B
            -2,  1, -2,
             1,  4,  1,
            -2,  1, -2,
        },
        { // detail C
            -1, -1, -1,
            -1,  4,  3,
            -1, -1, -1,
        },
        { // detail D
            -1, -1, -1,
            -1,  4, -1,
            -1, -1,  3,
        },

        { // line A
            -1, -1, -1,
             2,  2,  2,
            -1, -1, -1,
        },
        //{ // line B
        //     0,  1,  0,
        //     0,  1,  0,
        //     0,  1,  0,
        //},
        { // line B
            -2,  1,  1,
            -2,  1,  1,
            -2,  1,  1,
        },
        //{ // line B
        //    -1,  2, -1,
        //    -1,  2, -1,
        //    -1,  2, -1,
        //},
        { // line C
             2, -1, -1,
            -1,  2, -1,
            -1, -1,  2,
        },
        { // line D
            -1, -1,  2,
            -1,  2, -1,
             2, -1, -1,
        },

    };

    // Uses Shared Memory for pixel read.
    //
    //__global__ void ATS2 (
    //    /* IN  */ cudaTextureObject_t   texture,
    //    /* OUT */ u8*                   output,
    //    /* IN  */ u16                   iWidth,
    //    /* IN  */ u16                   iHeight,
    //    /* IN  */ u16                   oWidth,
    //    /* IN  */ u16                   oHeight
    //) {
    //    const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    //    const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
//
//
    //    // Calculate position in output array.
    //    //
    //    //const u8 channels = 4;
    //    //u32 id = (y * oWidth + x) * channels;
//
//
    //    //  Shared Memory.
    //    //
    //    __shared__ uchar4 blockPixelGrid [(THREADS_16 + 2) * (THREADS_16 + 2)];
    //    const u32 threadId = threadIdx.y * blockDim.x + threadIdx.x;
    //    const u8 pixelGridWidth = (THREADS_16 + 2);
    //    const u32 bx = threadIdx.x + 1;
    //    const u32 by = threadIdx.y + 1;
//
//
    //    // Make it so 68 of 16x16 threads also calculate a pixel of 16x16 grid halo.
    //    //
    //    if (threadId < 68) {
//
    //        // {  UP  } [ 1 -> 16] 
    //        if (threadId < 16) {                
    //            r32 u = ((r32)x + 0.5f) / iWidth;
    //            r32 v = ((r32)y - 0.5f) / iHeight;
    //            blockPixelGrid[((by - 1) * pixelGridWidth) + bx] = tex2D<uchar4> (texture, u, v);
    //        }
//
    //        // { LEFT } [(0 -> 17) * 18]
    //        else if (threadId < 16 + (18 * 1)) { 
    //            u16 idv = threadId - 16;
    //        
    //            r32 u = ((r32)x - 0.5f)       / iWidth;
    //            r32 v = ((r32)y + idv - 0.5f) / iHeight;
    //        
    //            blockPixelGrid[(idv * pixelGridWidth) + 0] = tex2D<uchar4> (texture, u, v);
    //        }
    //        
    //        // { RGHT } [ (0 -> 17) * 18 + 17]
    //        else if (threadId < 16 + (18 * 2)) { 
    //            u16 idv = threadId - (16 + 18);
    //        
    //            r32 u = ((r32)x + 17.5f)      / iWidth;
    //            r32 v = ((r32)y + idv - 0.5f) / iHeight;
    //        
    //            blockPixelGrid[(idv * pixelGridWidth) + 17] = tex2D<uchar4> (texture, u, v);
    //        }
//
    //        // { DOWN } [ 306 -> 322]
    //        else {                              
    //            r32 u = ((r32)x + 0.5f) / iWidth;
    //            r32 v = ((r32)y + 1.5f) / iHeight;
    //            blockPixelGrid[((by + 1) * pixelGridWidth) + bx] = tex2D<uchar4> (texture, u, v);
    //        }
//
    //    }
//
//
    //    { // Get a pixel from 16x16 grid.
    //        r32 u = ((r32)x + 0.5f) / iWidth;
    //        r32 v = ((r32)y + 0.5f) / iHeight;
    //        blockPixelGrid[(by * pixelGridWidth) + bx] = tex2D<uchar4> (texture, u, v);
    //    }
    //    
//
    //    // Synchronize threads. (Some threads do more then the rest.)
    //    //
    //    __syncthreads ();
//
//
    //    // The 3x3 pixel group.
    //    //blockPixelGrid[((by - 1) * pixelGridWidth) + bx - 1];
    //    //blockPixelGrid[((by - 1) * pixelGridWidth) + bx + 0];
    //    //blockPixelGrid[((by - 1) * pixelGridWidth) + bx + 1];
    //    //
    //    //blockPixelGrid[((by + 0) * pixelGridWidth) + bx - 1];
    //    //blockPixelGrid[((by + 0) * pixelGridWidth) + bx + 0];
    //    //blockPixelGrid[((by + 0) * pixelGridWidth) + bx + 1];
    //    //
    //    //blockPixelGrid[((by + 1) * pixelGridWidth) + bx - 1];
    //    //blockPixelGrid[((by + 1) * pixelGridWidth) + bx + 0];
    //    //blockPixelGrid[((by + 1) * pixelGridWidth) + bx + 1];
//
    //}


    __global__ void ATS1 (
        /* IN  */ cudaTextureObject_t   texture,
        /* OUT */ u8*                   output,
        /* IN  */ u16                   iWidth,
        /* IN  */ u16                   iHeight,
        /* IN  */ u16                   oWidth,
        /* IN  */ u16                   oHeight
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;


        // Calculate position in output array.
        //
        const u8 channels = 4;
        u32 id = (y * oWidth + x) * channels;

        uchar4 mainPixel;
        s16 r, g, b;

        { // main
            r32 u = ((r32)x + 0.5f) / iWidth;
            r32 v = ((r32)y + 0.5f) / iHeight;
            mainPixel = tex2D<uchar4> (texture, u, v);

            r = mainPixel.x;
            g = mainPixel.y;
            b = mainPixel.z;
        }


        { // horizontal
            { // left
                r32 u = ((r32)x - 0.5f) / iWidth;
                r32 v = ((r32)y + 0.5f) / iHeight;
                uchar4 pixel = tex2D<uchar4> (texture, u, v);
        
                r = pixel.x;
                g = pixel.y;
                b = pixel.z;
            }
        
            { // right
                r32 u = ((r32)x + 1.5f) / iWidth;
                r32 v = ((r32)y + 0.5f) / iHeight;
                uchar4 pixel = tex2D<uchar4> (texture, u, v);
        
                r += pixel.x;
                g += pixel.y;
                b += pixel.z;
            }
        }


        // { // vertical
        //     { // down
        //         r32 u = ((r32)x + 0.5f) / iWidth;
        //         r32 v = ((r32)y - 0.5f) / iHeight;
        //         uchar4 pixel = tex2D<uchar4> (texture, u, v);
        // 
        //         r = pixel.x;
        //         g = pixel.y;
        //         b = pixel.z;
        //     }
        // 
        //     { // up
        //         r32 u = ((r32)x + 0.5f) / iWidth;
        //         r32 v = ((r32)y + 1.5f) / iHeight;
        //         uchar4 pixel = tex2D<uchar4> (texture, u, v);
        // 
        //         r += pixel.x;
        //         g += pixel.y;
        //         b += pixel.z;
        //     }
        // }


        {
            // normalize
            r /= 2;
            g /= 2;
            b /= 2;

            // difference
            r = abs ((s16)mainPixel.x - r);
            g = abs ((s16)mainPixel.y - g);
            b = abs ((s16)mainPixel.z - b);
        }


        if ((r + g + b) < 8) {
            output[id + 0] = (u8) mainPixel.x;
            output[id + 1] = (u8) mainPixel.y;
            output[id + 2] = (u8) mainPixel.z;
            output[id + 3] = 255;
        } else {
            output[id + 0] = (u8) 0;
            output[id + 1] = (u8) 0;
            output[id + 2] = (u8) 0;
            output[id + 3] = 255;
        }


        //#pragma unroll
        //for (u8 n = 0; n < 3; ++n) { // Horiznontal Line
        //    r32 u = ((r32)x + n - 1.0f + 0.5f) / iWidth;
        //    r32 v = ((r32)y + 0.5f) / iHeight;
        //
        //    mainPixel = tex2D<uchar4> (texture, u, v);
        //}
        //output[id + 0] = (u8) mainPixel.x;
        //output[id + 1] = (u8) mainPixel.y;
        //output[id + 2] = (u8) mainPixel.z;
        //output[id + 3] = 255;
    }

}


namespace EDGE::TEXTURE::FILTERING {

    // Main + Corners
    //
    __global__ void CustomA (
        /* IN  */ cudaTextureObject_t   texture,
        /* OUT */ u8*                   output,
        /* IN  */ u16                   iWidth,
        /* IN  */ u16                   iHeight,
        /* IN  */ u16                   oWidth,
        /* IN  */ u16                   oHeight
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

        // Calculate position in output array.
        //
        const u8 channels = 4;
        u32 id = (y * oWidth + x) * channels;

        // Partial: [output space] -> [input space] (with fraction).
        //  2x -> -0.25, +0.25
        r32 px = (x + 0.5f) * ((r32) iWidth / oWidth)  ; 
        r32 py = (y + 0.5f) * ((r32) iHeight / oHeight);

        // Integral part (pixel in input image).
        //
        auto ix = (u16) px;
        auto iy = (u16) py;

        // Extract fraction part (in-pixel position in input image).
        //
        r32 fx = px - ix;
        r32 fy = py - iy;

        // A pixel field.
        // [-1,+0,+1,+2]
        // [+0,  ,  ,  ]
        // [+1,  ,  ,  ]
        // [+2,  ,  ,  ]
        //
        float4 patch[3][3];
        float4 result;

        // Get said field of pixels from the input texture.
        //
        for (u8 m = 0; m < 3; ++m) {
            for (u8 n = 0; n < 3; ++n) {

                const r32 fn = ((r32)n) - 1.0f;
                const r32 fm = ((r32)m) - 1.0f;

                r32 u = ((((r32) ix) + fn + 0.5f) / iWidth);
                r32 v = ((((r32) iy) + fm + 0.5f) / iHeight);

                uchar4 pixel = tex2D<uchar4> (texture, u, v);

                patch[m][n].x = ((r32)pixel.x) / 255.0f;
                patch[m][n].y = ((r32)pixel.y) / 255.0f;
                patch[m][n].z = ((r32)pixel.z) / 255.0f;
                patch[m][n].w = ((r32)pixel.w) / 255.0f;
            }
        }

        {
            const r32 mainPixel = 0.5f;
            const r32 k1 = (r32) (fx >= 0.5f) * (fy >= 0.5f) * 0.5f;
            const r32 k2 = (r32) (fx >= 0.5f) * (fy <  0.5f) * 0.5f;
            const r32 k3 = (r32) (fx <  0.5f) * (fy >= 0.5f) * 0.5f;
            const r32 k4 = (r32) (fx <  0.5f) * (fy <  0.5f) * 0.5f;
            //
            result.x = (mainPixel * patch[1][1].x) + 
                (k4 * patch[0][0].x) +  // left-up
                (k2 * patch[0][2].x) +  // up-right
                (k1 * patch[2][2].x) +  // right-down
                (k3 * patch[2][0].x);   // down-left
            
            result.y = (mainPixel * patch[1][1].y) + 
                (k4 * patch[0][0].y) +  // left
                (k2 * patch[0][2].y) +  // up
                (k1 * patch[2][2].y) +  // right
                (k3 * patch[2][0].y);   // down
            
            result.z = (mainPixel * patch[1][1].z) + 
                (k4 * patch[0][0].z) +  // left
                (k2 * patch[0][2].z) +  // up
                (k1 * patch[2][2].z) +  // right
                (k3 * patch[2][0].z);   // down
            
            result.w = (mainPixel * patch[1][1].w) + 
                (k4 * patch[0][0].w) +  // left
                (k2 * patch[0][2].w) +  // up
                (k1 * patch[2][2].w) +  // right
                (k3 * patch[2][0].w) ;  // down
        }

        // Emplace the result.
        //
        output[id + 0] = (u8) min (max (result.x * 255.0f, 0.0f), 255.0f);
        output[id + 1] = (u8) min (max (result.y * 255.0f, 0.0f), 255.0f);
        output[id + 2] = (u8) min (max (result.z * 255.0f, 0.0f), 255.0f);
        output[id + 3] = 255;
    }


    // Main + Sides
    //
    __global__ void CustomB (
        /* IN  */ cudaTextureObject_t   texture,
        /* OUT */ u8*                   output,
        /* IN  */ u16                   iWidth,
        /* IN  */ u16                   iHeight,
        /* IN  */ u16                   oWidth,
        /* IN  */ u16                   oHeight
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

        // Calculate position in output array.
        //
        const u8 channels = 4;
        u32 id = (y * oWidth + x) * channels;

        // Partial: [output space] -> [input space] (with fraction).
        //  2x -> -0.25, +0.25
        r32 px = (x + 0.5f) * ((r32) iWidth / oWidth)  ; 
        r32 py = (y + 0.5f) * ((r32) iHeight / oHeight);

        // Integral part (pixel in input image).
        //
        auto ix = (u16) px;
        auto iy = (u16) py;

        // Extract fraction part (in-pixel position in input image).
        //
        r32 fx = px - ix;
        r32 fy = py - iy;

        // A pixel field.
        // [-1,+0,+1,+2]
        // [+0,  ,  ,  ]
        // [+1,  ,  ,  ]
        // [+2,  ,  ,  ]
        //
        float4 patch[3][3];
        float4 result;

        // Get said field of pixels from the input texture.
        //
        for (u8 m = 0; m < 3; ++m) {
            for (u8 n = 0; n < 3; ++n) {

                const r32 fn = ((r32)n) - 1.0f;
                const r32 fm = ((r32)m) - 1.0f;

                r32 u = ((((r32) ix) + fn + 0.5f) / iWidth);
                r32 v = ((((r32) iy) + fm + 0.5f) / iHeight);

                uchar4 pixel = tex2D<uchar4> (texture, u, v);

                patch[m][n].x = ((r32)pixel.x) / 255.0f;
                patch[m][n].y = ((r32)pixel.y) / 255.0f;
                patch[m][n].z = ((r32)pixel.z) / 255.0f;
                patch[m][n].w = ((r32)pixel.w) / 255.0f;
            }
        }

        {
            const r32 mainPixel = 0.5f;
            const r32 ix = (r32) (fx >= 0.5f) * 0.25f;
            const r32 iy = (r32) (fy >= 0.5f) * 0.25f;
            const r32 jx = (r32) (fx <  0.5f) * 0.25f;
            const r32 jy = (r32) (fy <  0.5f) * 0.25f;

            result.x = (mainPixel * patch[1][1].x) + 
                (jx * patch[1][0].x) +  // left
                (jy * patch[0][1].x) +  // up
                (ix * patch[1][2].x) +  // right
                (iy * patch[2][1].x);   // down
            
            result.y = (mainPixel * patch[1][1].y) + 
                (jx * patch[1][0].y) +  // left
                (jy * patch[0][1].y) +  // up
                (ix * patch[1][2].y) +  // right
                (iy * patch[2][1].y);   // down
            
            result.z = (mainPixel * patch[1][1].z) + 
                (jx * patch[1][0].z) +  // left
                (jy * patch[0][1].z) +  // up
                (ix * patch[1][2].z) +  // right
                (iy * patch[2][1].z);   // down
            
            result.w = (mainPixel * patch[1][1].w) + 
                (jx * patch[1][0].w) +  // left
                (jy * patch[0][1].w) +  // up
                (ix * patch[1][2].w) +  // right
                (iy * patch[2][1].w) ;  // down
        }

        // Emplace the result.
        //
        output[id + 0] = (u8) min (max (result.x * 255.0f, 0.0f), 255.0f);
        output[id + 1] = (u8) min (max (result.y * 255.0f, 0.0f), 255.0f);
        output[id + 2] = (u8) min (max (result.z * 255.0f, 0.0f), 255.0f);
        output[id + 3] = 255;
    }

}


namespace EDGE::TEXTURE::FILTERING { 

    // Used for calculating Bicubic filtering.
    //  A different approach is also possible.
    __device__ r32 CatmullRomSpline (
        /* IN  */ const r32& p0, // -1 element 
        /* IN  */ const r32& p1, //  0 element
        /* IN  */ const r32& p2, //  1 element
        /* IN  */ const r32& p3, //  2 element
        /* IN  */ const r32& t   //  [0-1] where on initial texture pixel are we - value.
    ) {
        r32 a0 = (p3 - p2) - (p0 - p1);
        r32 a1 = (p0 - p1) - a0;
        r32 a2 = p2 - p0;
        r32 a3 = p1;
        
        r32 final = (a0 * t * t * t) + (a1 * t * t) + (a2 * t) + a3;

        return fminf (fmaxf (final, 0.0f), 1.0f);
    }

    // Bicubic filtering with Catmull-Rom can overshoot, causing negative values near edges.
    //
    __global__ void BicubicRW (
        /* IN  */ cudaTextureObject_t   texture,
        #ifdef ATS_ENABLE_RENDER_TO_FILE
        /* OUT */ u8*                   output,
        #else
        /* OUT */ cudaSurfaceObject_t   output,
        #endif
        /* IN  */ u16                   iWidth,
        /* IN  */ u16                   iHeight,
        /* IN  */ u16                   oWidth,
        /* IN  */ u16                   oHeight
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

        #ifdef ATS_ENABLE_RENDER_TO_FILE
            // Calculate position in output array.
            //
            const u8 channels = 4;
            u32 id = (y * oWidth + x) * channels;
        #endif

        // Partial: [output space] -> [input space] (with fraction).
        //  2x -> -0.25, +0.25
        r32 px = (x + 0.5f) * ((r32) iWidth / oWidth)     - 0.5f; 
        r32 py = (y + 0.5f) * ((r32) iHeight / oHeight)   - 0.5f;

        // Integral part (pixel in input image).
        //
        auto ix = (u16) px;
        auto iy = (u16) py;

        // Extract fraction part (in-pixel position in input image).
        //
        r32 fx = px - ix;
        r32 fy = py - iy;

        // A pixel field.
        // [-1,+0,+1,+2]
        // [+0,  ,  ,  ]
        // [+1,  ,  ,  ]
        // [+2,  ,  ,  ]
        //
        float4 patch[4][4]; 
        float4 col[4];
        float4 result;

        // Get said field of pixels from the input texture.
        //
        for (u8 m = 0; m < 4; ++m) {
            for (u8 n = 0; n < 4; ++n) {

                const r32 fn = ((r32)n) - 1.0f;
                const r32 fm = ((r32)m) - 1.0f;

                r32 u = ((((r32) ix) + fn + 0.5f) / iWidth);
                r32 v = ((((r32) iy) + fm + 0.5f) / iHeight);

                uchar4 pixel = tex2D<uchar4> (texture, u, v);

                patch[m][n].x = ((r32)pixel.x) / 255.0f;
                patch[m][n].y = ((r32)pixel.y) / 255.0f;
                patch[m][n].z = ((r32)pixel.z) / 255.0f;
                patch[m][n].w = ((r32)pixel.w) / 255.0f;
            }
        }

        // Calculate grid into a column.
        //
        for (u8 i = 0; i < 4; ++i) {
            col[i].x = CatmullRomSpline (patch[i][0].x, patch[i][1].x, patch[i][2].x, patch[i][3].x, fx);
            col[i].y = CatmullRomSpline (patch[i][0].y, patch[i][1].y, patch[i][2].y, patch[i][3].y, fx);
            col[i].z = CatmullRomSpline (patch[i][0].z, patch[i][1].z, patch[i][2].z, patch[i][3].z, fx);
            col[i].w = CatmullRomSpline (patch[i][0].w, patch[i][1].w, patch[i][2].w, patch[i][3].w, fx);
        }
        
        // Calculate column into a single value.
        //    
        result.x = CatmullRomSpline (col[0].x, col[1].x, col[2].x, col[3].x, fy);
        result.y = CatmullRomSpline (col[0].y, col[1].y, col[2].y, col[3].y, fy);
        result.z = CatmullRomSpline (col[0].z, col[1].z, col[2].z, col[3].z, fy);
        result.w = CatmullRomSpline (col[0].w, col[1].w, col[2].w, col[3].w, fy);

        //
        // Emplace the result.
        //

        #ifdef ATS_ENABLE_RENDER_TO_FILE
            output[id + 0] = (u8) min (max (result.x * 255.0f, 0.0f), 255.0f);
            output[id + 1] = (u8) min (max (result.y * 255.0f, 0.0f), 255.0f);
            output[id + 2] = (u8) min (max (result.z * 255.0f, 0.0f), 255.0f);
            output[id + 3] = 255;
        #else
            uchar4 pixel {
                (u8) min (max (result.x * 255.0f, 0.0f), 255.0f),
                (u8) min (max (result.y * 255.0f, 0.0f), 255.0f),
                (u8) min (max (result.z * 255.0f, 0.0f), 255.0f),
                255
            };

            surf2Dwrite<uchar4> (pixel, output, x * sizeof (uchar4), y);
        #endif

    }


    // Bicubic filtering with Catmull-Rom can overshoot, causing negative values near edges.
    //
    __global__ void Bicubic (
        /* IN  */ cudaTextureObject_t   texture,
        /* OUT */ u8*                   output,
        /* IN  */ u16                   iWidth,
        /* IN  */ u16                   iHeight,
        /* IN  */ u16                   oWidth,
        /* IN  */ u16                   oHeight
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

        // Calculate position in output array.
        //
        const u8 channels = 4;
        u32 id = (y * oWidth + x) * channels;

        // Partial: [output space] -> [input space] (with fraction).
        //  2x -> -0.25, +0.25
        r32 px = (x + 0.5f) * ((r32) iWidth / oWidth)     - 0.5f; 
        r32 py = (y + 0.5f) * ((r32) iHeight / oHeight)   - 0.5f;

        // Integral part (pixel in input image).
        //
        auto ix = (u16) px;
        auto iy = (u16) py;

        // Extract fraction part (in-pixel position in input image).
        //
        r32 fx = px - ix;
        r32 fy = py - iy;

        // A pixel field.
        // [-1,+0,+1,+2]
        // [+0,  ,  ,  ]
        // [+1,  ,  ,  ]
        // [+2,  ,  ,  ]
        //
        float4 patch[4][4]; 
        float4 col[4];
        float4 result;

        // Get said field of pixels from the input texture.
        //
        for (u8 m = 0; m < 4; ++m) {
            for (u8 n = 0; n < 4; ++n) {

                const r32 fn = ((r32)n) - 1.0f;
                const r32 fm = ((r32)m) - 1.0f;

                r32 u = ((((r32) ix) + fn + 0.5f) / iWidth);
                r32 v = ((((r32) iy) + fm + 0.5f) / iHeight);

                uchar4 pixel = tex2D<uchar4> (texture, u, v);

                patch[m][n].x = ((r32)pixel.x) / 255.0f;
                patch[m][n].y = ((r32)pixel.y) / 255.0f;
                patch[m][n].z = ((r32)pixel.z) / 255.0f;
                patch[m][n].w = ((r32)pixel.w) / 255.0f;
            }
        }

        // Calculate grid into a column.
        //
        for (u8 i = 0; i < 4; ++i) {
            col[i].x = CatmullRomSpline (patch[i][0].x, patch[i][1].x, patch[i][2].x, patch[i][3].x, fx);
            col[i].y = CatmullRomSpline (patch[i][0].y, patch[i][1].y, patch[i][2].y, patch[i][3].y, fx);
            col[i].z = CatmullRomSpline (patch[i][0].z, patch[i][1].z, patch[i][2].z, patch[i][3].z, fx);
            col[i].w = CatmullRomSpline (patch[i][0].w, patch[i][1].w, patch[i][2].w, patch[i][3].w, fx);
        }
        
        // Calculate column into a single value.
        //    
        result.x = CatmullRomSpline (col[0].x, col[1].x, col[2].x, col[3].x, fy);
        result.y = CatmullRomSpline (col[0].y, col[1].y, col[2].y, col[3].y, fy);
        result.z = CatmullRomSpline (col[0].z, col[1].z, col[2].z, col[3].z, fy);
        result.w = CatmullRomSpline (col[0].w, col[1].w, col[2].w, col[3].w, fy);

        // Emplace the result.
        //
        output[id + 0] = (u8) min (max (result.x * 255.0f, 0.0f), 255.0f);
        output[id + 1] = (u8) min (max (result.y * 255.0f, 0.0f), 255.0f);
        output[id + 2] = (u8) min (max (result.z * 255.0f, 0.0f), 255.0f);
        output[id + 3] = 255;
    }

}


namespace EDGE::TEXTURE::FILTER {


    enum KIRSCH_DIRECTION: u8 {
        KIRSCH_DIRECTION_N  = 0,
        KIRSCH_DIRECTION_NE = 1,
        KIRSCH_DIRECTION_E  = 2,
        KIRSCH_DIRECTION_SE = 3,
        KIRSCH_DIRECTION_S  = 4,
        KIRSCH_DIRECTION_SW = 5,
        KIRSCH_DIRECTION_W  = 6,
        KIRSCH_DIRECTION_NW = 7,
    };


    // Kirsch masks
    __device__ const s8 KIRSCH_MASKS [8][3][3] {
        {{  5,  5,  5 }, { -3, 0, -3}, { -3, -3, -3}}, // N
        {{  5,  5, -3 }, {  5, 0, -3}, { -3, -3, -3}}, // NE
        {{  5, -3, -3 }, {  5, 0, -3}, {  5, -3, -3}}, // E
        {{ -3, -3, -3 }, {  5, 0, -3}, {  5,  5, -3}}, // SE
        {{ -3, -3, -3 }, { -3, 0, -3}, {  5,  5,  5}}, // S
        {{ -3, -3, -3 }, { -3, 0,  5}, { -3,  5,  5}}, // SW
        {{ -3, -3,  5 }, { -3, 0,  5}, { -3, -3,  5}}, // W
        {{ -3,  5,  5 }, { -3, 0,  5}, { -3, -3, -3}}  // NW
    };


    // Kirsch filter. Produces thick, high-contrast edges. Strong directional sensitivity (more than Sobel).
    //  More computationally expensive due to 8 convolutions per pixel. Sensitive to noise, but 
    //  great for structured edge detection.
    //
    __global__ void KirschAll (
        /* IN  */ cudaTextureObject_t   texture,
        /* OUT */ u8*                   output,
        /* IN  */ u16                   width,
        /* IN  */ u16                   height
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

        s32 maxResponse = 0;

        for (u8 k = 0; k < 8; ++k) {
            r32 sum = 0;

            for (u8 j = 0; j < 3; ++j) {
                for (u8 i = 0; i < 3; ++i) {

                    r32 u = (x + i - 1.0f + 0.5f) / width;
                    r32 v = (y + j - 1.0f + 0.5f) / height;

                    uchar4 pixel = tex2D<uchar4> (texture, u, v);
                    r32 gs = GRAYSCALE (pixel.x, pixel.y, pixel.z);

                    sum += gs * KIRSCH_MASKS[k][j][i];
                }
            }

            maxResponse = max (maxResponse, abs ((s32) sum));
        }

        // Clamp to [0, 255].
        //
        u8 intensity = min (255, maxResponse);

        output[(y * width * 4) + (x * 4) + 0] = intensity;
        output[(y * width * 4) + (x * 4) + 1] = intensity;
        output[(y * width * 4) + (x * 4) + 2] = intensity;
        output[(y * width * 4) + (x * 4) + 3] = 255;
    }

    // Kirsch filter. Same but only appled on a single  selected direction.
    // 
    __global__ void KirschSingle (
        /* IN  */ cudaTextureObject_t   texture,
        /* OUT */ u8*                   output,
        /* IN  */ u16                   width,
        /* IN  */ u16                   height,
        /* IN  */ u8                    direction
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

        s32 maxResponse = 0;
        r32 sum = 0;

        for (u8 j = 0; j < 3; ++j) {
            for (u8 i = 0; i < 3; ++i) {

                r32 u = (x + i - 1.0f + 0.5f) / width;
                r32 v = (y + j - 1.0f + 0.5f) / height;

                uchar4 pixel = tex2D<uchar4> (texture, u, v);
                r32 gs = GRAYSCALE(pixel.x, pixel.y, pixel.z);

                sum += gs * KIRSCH_MASKS[direction][j][i];
            }
        }

        maxResponse = max (maxResponse, abs ((s32) sum));

        // Clamp to [0, 255].
        //
        u8 intensity = min (255, maxResponse);

        output[(y * width * 4) + (x * 4) + 0] = intensity;
        output[(y * width * 4) + (x * 4) + 1] = intensity;
        output[(y * width * 4) + (x * 4) + 2] = intensity;
        output[(y * width * 4) + (x * 4) + 3] = 255;
    }


    // Laplacian filter. Computes the second derivative which is where that change accelerate.
    // It's isotropic (direction-independent), sharp changes, center-emphasized, does not smooth.
    // Creates a white outline and a black inner edge
    //
    __global__ void LaplacianA (
        /* IN  */ cudaTextureObject_t   texture,
        /* OUT */ u8*                   output,
        /* IN  */ u16                   width,
        /* IN  */ u16                   height
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
    
        // We access the texture pixels using 1/length formula. (INVERSE MAPPING)
        r32 u = (r32) ( x + 0.5f ) / width;
        r32 v = (r32) ( y + 0.5f ) / height;
    
        //uchar4 pixel = tex2D<uchar4> (texture, u, v);

        {
            r32 du = 1.0f / width;
            r32 dv = 1.0f / height;
        
            // TODO `fmaxf` vs `max`
            uchar4 lt = tex2D<uchar4> (texture, max (u - du, 0.0f), v);
            uchar4 rt = tex2D<uchar4> (texture, min (u + du, 1.0f), v);
            uchar4 up = tex2D<uchar4> (texture, u, max (v - dv, 0.0f));
            uchar4 dn = tex2D<uchar4> (texture, u, min (v + dv, 1.0f));
        
            // r32 gsPixel = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
            r32 gsLt = GRAYSCALE (lt.x, lt.y, lt.z);
            r32 gsRt = GRAYSCALE (rt.x, rt.y, rt.z);
            r32 gsUp = GRAYSCALE (up.x, up.y, up.z);
            r32 gsDn = GRAYSCALE (dn.x, dn.y, dn.z);
        
            r32 dx = gsRt - gsLt;
            r32 dy = gsDn - gsUp;
            r32 edge = sqrtf (dx * dx + dy * dy); 
        
            // Edge is in 0-255 range so we can easily convert it to byte.
            u8 intensity = (u8) edge;
            
            output[(y * width * 4) + (x * 4) + 0] = intensity;
            output[(y * width * 4) + (x * 4) + 1] = intensity;
            output[(y * width * 4) + (x * 4) + 2] = intensity;
            output[(y * width * 4) + (x * 4) + 3] = 255;
        }
    }


    // Laplacian. Uses an actual KERNEL. TODO. It might be better to use input image only 
    //  then process it into output.
    //
    __global__ void LaplacianB (
        /* IN  */ cudaTextureObject_t   texture,
        /* OUT */ u8*                   output,
        /* IN  */ u16                   iWidth,
        /* IN  */ u16                   iHeight,
        /* IN  */ u16                   oWidth,
        /* IN  */ u16                   oHeight
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

        s8 kernel[3][3] {
            { -1, -1, -1 },
            { -1,  8, -1 },
            { -1, -1, -1 }
        };

        r32 intensity = 0;

        for (u8 m = 0; m < 3; m++) {
            for (u8 n = 0; n < 3; n++) {

                r32 u = ((r32)x + n - 1.0f + 0.5f) / iWidth;
                r32 v = ((r32)y + m - 1.0f + 0.5f) / iHeight;

                uchar4 pixel = tex2D<uchar4> (texture, u, v);

                r32 gs = GRAYSCALE (pixel.x, pixel.y, pixel.z);

                intensity += gs * kernel[m][n];
            }
        }

        // Not needed
        //intensity = max (0, min (255, intensity));

        output[(y * oWidth * 4) + (x * 4) + 0] = intensity;
        output[(y * oWidth * 4) + (x * 4) + 1] = intensity;
        output[(y * oWidth * 4) + (x * 4) + 2] = intensity;
        output[(y * oWidth * 4) + (x * 4) + 3] = 255;

    }


    // Sobel filter
    // Performs edge detection by sampling neighbors and applying an edge filter. 
    //  It’s simple and fast, good for real-time use. It smooths a little.
    // -> Gradient magnitude approach.
    // -> changes in intensity, which correspond to edges.
    // 
    __global__ void Sobel (
        /* IN  */ cudaTextureObject_t   texture,
        /* OUT */ u8*                   output,
        /* IN  */ u16                   width,
        /* IN  */ u16                   height
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
    
        // We access the texture pixels using 1/length formula. (INVERSE MAPPING)
        r32 u = (r32) ( x + 0.5f ) / width;
        r32 v = (r32) ( y + 0.5f ) / height;
    
        uchar4 pixel = tex2D<uchar4> (texture, u, v);

        {
            r32 du = 1.0f / width;
            r32 dv = 1.0f / height;
            
            float2 offsets [9] {
                { -du,  dv   }, { 0.0f,  dv   }, { du,  dv   },
                { -du,  0.0f }, { 0.0f,  0.0f }, { du,  0.0f },
                { -du, -dv   }, { 0.0f, -dv   }, { du, -dv   },
            };
        
            r32 kernel[9] { 
                1, 1, 1,
                1,-8, 1,
                1, 1, 1, 
            };
        
            float3 col = { 0, 0, 0 };
        
            for (u8 i = 0; i < 9; ++i) {
                uchar4 image = tex2D<uchar4> (texture, u + offsets[i].x, v + offsets[i].y);

                col.x += image.x * kernel[i];
                col.y += image.y * kernel[i];
                col.z += image.z * kernel[i];
            }
        
            { // Proper taking care of float - byte color conversion.
                // 1. Take absolute value — typical for edge detection visualizations
                col.x = fabsf (col.x);
                col.y = fabsf (col.y);
                col.z = fabsf (col.z);
        
                // 2. Normalize (tweak scale as needed for smoothness)
                r32 scale = 1.0f / 8.0f;
                col.x *= scale;
                col.y *= scale;
                col.z *= scale;
        
                // 3. Clamp to 0–255
                col.x = fminf (fmaxf (col.x, 0.0f), 255.0f);
                col.y = fminf (fmaxf (col.y, 0.0f), 255.0f);
                col.z = fminf (fmaxf (col.z, 0.0f), 255.0f);
            }
        
            output[(y * width * 4) + (x * 4) + 0] = (u8) col.x;
            output[(y * width * 4) + (x * 4) + 1] = (u8) col.y;
            output[(y * width * 4) + (x * 4) + 2] = (u8) col.z;
            output[(y * width * 4) + (x * 4) + 3] = 255;
        }
    }

}


namespace EDGE::TEXTURE::FORWARD {

    __global__ void Fill2RGB (
        /* IN  */ cudaTextureObject_t   texture,
        /* OUT */ u8*                   output,
        /* IN  */ u16                   width,
        /* IN  */ u16                   height
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

        uchar4 pixel = tex2D<uchar4> (texture, x, y);

        const auto channels = 3;
        const auto scale = 2;

        const auto row = width * channels;
        const auto col = channels;

        const auto idx = (x * col * scale);
        const auto idy = (y * row * scale);

        output[idy + (row * 0) + idx + (col * 0) + 0] = pixel.x;
        output[idy + (row * 0) + idx + (col * 0) + 1] = pixel.y;
        output[idy + (row * 0) + idx + (col * 0) + 2] = pixel.z;

        output[idy + (row * 0) + idx + (col * 1) + 0] = pixel.x;
        output[idy + (row * 0) + idx + (col * 1) + 1] = pixel.y;
        output[idy + (row * 0) + idx + (col * 1) + 2] = pixel.z;

        output[idy + (row * 1) + idx + (col * 0) + 0] = pixel.x;
        output[idy + (row * 1) + idx + (col * 0) + 1] = pixel.y;
        output[idy + (row * 1) + idx + (col * 0) + 2] = pixel.z;

        output[idy + (row * 1) + idx + (col * 1) + 0] = pixel.x;
        output[idy + (row * 1) + idx + (col * 1) + 1] = pixel.y;
        output[idy + (row * 1) + idx + (col * 1) + 2] = pixel.z;
    }


    __global__ void Fill2RGBA (
        /* IN  */ cudaTextureObject_t   texture,
        /* OUT */ u8*                   output,
        /* IN  */ u16                   width,
        /* IN  */ u16                   height
    ) {
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

        uchar4 pixel = tex2D<uchar4> (texture, x, y);

        const auto channels = 4;
        const auto scale = 2;

        const auto row = width * channels;
        const auto col = channels;

        const auto idx = (x * col * scale);
        const auto idy = (y * row * scale);

        output[idy + (row * 0) + idx + (col * 0) + 0] = pixel.x;
        output[idy + (row * 0) + idx + (col * 0) + 1] = pixel.y;
        output[idy + (row * 0) + idx + (col * 0) + 2] = pixel.z;
        output[idy + (row * 0) + idx + (col * 0) + 3] = 255;

        output[idy + (row * 0) + idx + (col * 1) + 0] = pixel.x;
        output[idy + (row * 0) + idx + (col * 1) + 1] = pixel.y;
        output[idy + (row * 0) + idx + (col * 1) + 2] = pixel.z;
        output[idy + (row * 0) + idx + (col * 1) + 3] = 255;

        output[idy + (row * 1) + idx + (col * 0) + 0] = pixel.x;
        output[idy + (row * 1) + idx + (col * 0) + 1] = pixel.y;
        output[idy + (row * 1) + idx + (col * 0) + 2] = pixel.z;
        output[idy + (row * 1) + idx + (col * 0) + 3] = 255;

        output[idy + (row * 1) + idx + (col * 1) + 0] = pixel.x;
        output[idy + (row * 1) + idx + (col * 1) + 1] = pixel.y;
        output[idy + (row * 1) + idx + (col * 1) + 2] = pixel.z;
        output[idy + (row * 1) + idx + (col * 1) + 3] = 255;
    }

}