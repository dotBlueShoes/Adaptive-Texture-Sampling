#pragma once
#include "framework.cuh"

#define GRAYSCALE_A(r, g, b) \
    (0.299f * r) + (0.587f * g) + (0.114f * b);

#define GRAYSCALE_B(r, g, b) \
    (0.21f * r) + (0.72f * g) + (0.07f * b);

#define GRAYSCALE_C(r, g, b) \
    (r + g + b) / 3;

#define GRAYSCALE(r, g, b) \
    GRAYSCALE_A (r, g, b)

namespace EDGE2 {

    //nope // Kirsch masks (excludes center)
    //nope __device__ const s8 KIRSCH_MASKS_4 [4][3][3] {
    //nope     {{  5,  5, -3 }, {  5, 0, -3}, { -3, -3, -3}}, // NE // LT
    //nope     {{ -3,  5,  5 }, { -3, 0,  5}, { -3, -3, -3}}, // NW // TR
    //nope     {{ -3, -3, -3 }, {  5, 0, -3}, {  5,  5, -3}}, // SE // BL
    //nope     {{ -3, -3, -3 }, { -3, 0,  5}, { -3,  5,  5}}  // SW // RB
    //nope };

    __global__ void Linear (
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

        r32 px = (x + 0.5f) * ((r32) iWidth / oWidth); 
        r32 py = (y + 0.5f) * ((r32) iHeight / oHeight);

        auto ix = (u16) px;
        auto iy = (u16) py;

        // 0.0 -> ~1.0
        r32 fx = px - ix;
        r32 fy = py - iy;

        uchar4 pixels [9];

        for (u8 yi = 0; yi < 3; ++yi) { // unroll loop - opt
            for (u8 xi = 0; xi < 3; ++xi) {
                r32 u = ((((r32) ix) + (xi) + (-1.0) + 0.5f) / iWidth);
                r32 v = ((((r32) iy) + (yi) + (-1.0) + 0.5f) / iHeight);

                pixels[(yi * 3) + xi] = tex2D<uchar4> (texture, u, v);
            }
        }

        auto& pixel = pixels[4];

        //  Space Transform
        // 0.0 <-> 1.0 to -0.5 <-> 0.5
        fx = fx - 0.5f;
        fy = fy - 0.5f;

        //  ABOUT
        // Get how strong left/right/up/down is. Set the opposite axis strength to 0.0.
        r32 cl = (fx < 0) ? -(fx) : 0.0f;
        r32 cr = (fx > 0) ?  (fx) : 0.0f;
        r32 ct = (fy < 0) ? -(fy) : 0.0f;
        r32 cb = (fy > 0) ?  (fy) : 0.0f;

        //
        // Those are not official linear wages.
        //

        r32 r1 = (cl + ct) * 0.25f / 3.0f;                     // -0.25 -> 0.15 (1/3 z 1/4)
        r32 r2 = ct        * (1.0f / 3.0f);                    // -0.25 -> 0.05 (1/3 z 1/4)
        r32 r3 = (ct + cr) * 0.25f / 3.0f;                     // -0.25 -> 0.00
        r32 r4 = cl        * (1.0f / 3.0f);                    // -0.25 -> 0.05 (1/3 z 1/4)
        r32 r5 = (1.0f - abs (fx) + 1.0f - abs (fy)) * 0.5f;   // -0.25 -> 0.75 (3/4)
        r32 r6 = cr        * (1.0f / 3.0f);                    // -0.25 -> 0.00
        r32 r7 = (cb + cl) * 0.25f / 3.0f;                     // -0.25 -> 0.00
        r32 r8 = cb        * (1.0f / 3.0f);                    // -0.25 -> 0.00
        r32 r9 = (cr + cb) * 0.25f / 3.0f;                     // -0.25 -> 0.00

        pixel.x = 
            r1 * pixels[0].x + r2 * pixels[1].x + r3 * pixels[2].x + 
            r4 * pixels[3].x + r5 * pixels[4].x + r6 * pixels[5].x + 
            r7 * pixels[6].x + r8 * pixels[7].x + r9 * pixels[8].x;

        pixel.y = 
            r1 * pixels[0].y + r2 * pixels[1].y + r3 * pixels[2].y + 
            r4 * pixels[3].y + r5 * pixels[4].y + r6 * pixels[5].y + 
            r7 * pixels[6].y + r8 * pixels[7].y + r9 * pixels[8].y;

        pixel.z = 
            r1 * pixels[0].z + r2 * pixels[1].z + r3 * pixels[2].z + 
            r4 * pixels[3].z + r5 * pixels[4].z + r6 * pixels[5].z + 
            r7 * pixels[6].z + r8 * pixels[7].z + r9 * pixels[8].z;

        #ifdef ATS_ENABLE_RENDER_TO_FILE
            output[id + 0] = pixel.x;
            output[id + 1] = pixel.y;
            output[id + 2] = pixel.z;
            output[id + 3] = 255;
        #else
            surf2Dwrite<uchar4> (pixel, output, x * sizeof (uchar4), y);
        #endif

    }

    __global__ void Nearest (
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

        r32 u = ((((r32) x) + 0.5f) / oWidth);
        r32 v = ((((r32) y) + 0.5f) / oHeight);

        uchar4 pixel = tex2D<uchar4> (texture, u, v);

        #ifdef ATS_ENABLE_RENDER_TO_FILE
            output[id + 0] = pixel.x;
            output[id + 1] = pixel.y;
            output[id + 2] = pixel.z;
            output[id + 3] = 255;
        #else
            surf2Dwrite<uchar4> (pixel, output, x * sizeof (uchar4), y);
        #endif

    }


    // NOTES
    // scalling raczej celujemy w ~2x ale fajnie by było jeśli algorytm będzie działał też do 3x i 4x
    // 5 direction system (lt, tr, rb, bl + none) 
    // vs
    // 7 direction system (+ horizontal + vertical)
    // mając jeszcze dodatkowo hor i ver bardziej dokładne będzie tworzenie skalowania dla 3x?
    //
    // CELE
    // 1. Utworzyć pixele (myśl jak nearest-neighbour) aby zachować linie po upscale
    // 2. Wygładzić linie aby efekt był bliższemu temu z Cycles
    // WAŻNE
    // W skalowaniu np. 1.5 Wszystkie pixele będą przesunięte. Nic już nie będzie równo na 0.5 (centrum piksela)
    // pytanie czy poza tworzeniem nowego pixela z 4 pixeli (pixel najbliższy i 3 pobliskie pixele) mogę inaczej utworzyć pixel?
    // wykrywanie krzywizny a jej przesyłanie


    __global__ void EdgeA (
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
        
        // 1. Make shifts on pixels based on edge detection
        // -> what kind of edge detection
        // -> how do i get if said pixel is shifted pixel or new pixel
        // -> fill new pixels
        // DONE

        // how do i add to this fxaa/cmaa/myaa ?

        // xy of output image
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

        #ifdef ATS_ENABLE_RENDER_TO_FILE
            // Calculate position in output array.
            //
            const u8 channels = 4;
            u32 id = (y * oWidth + x) * channels;
        #endif

        // xy of output pixel in input space.
        r32 px = (x + 0.5f) * ((r32) iWidth / oWidth); 
        r32 py = (y + 0.5f) * ((r32) iHeight / oHeight);

        // integral part (xy of input image).
        auto ix = (u16) px;
        auto iy = (u16) py;

        // floating point part.
        r32 fx = px - ix;
        r32 fy = py - iy;

        uchar4 patch[9];
        r32 gss[9];

        for (u8 yi = 0; yi < 3; ++yi) {
            for (u8 xi = 0; xi < 3; ++xi) {

                // xy of input image in uv coordinates (a uv point is a normalized pixel point - divided by length)
                r32 iu = ((((r32) ix + xi) + (-1.0) + 0.5f) / iWidth);
                r32 iv = ((((r32) iy + yi) + (-1.0) + 0.5f) / iHeight);

                auto& pixel = patch[(yi * 3) + xi];
                auto& gs = gss[(yi * 3) + xi]; 

                pixel = tex2D<uchar4> (texture, iu, iv);
                gs = GRAYSCALE (pixel.x, pixel.y, pixel.z);

            }
        }

        r32 edgeTemps[4];

        ///nopefor (u8 edge = 0; edge < 4; ++edge) {
        ///nope    auto& edgeTemp = edgeTemps[edge];
        ///nope    r32 sum = 0;
        ///nope
        ///nope    for (u8 yi = 0; yi < 3; ++yi) {
        ///nope        for (u8 xi = 0; xi < 3; ++xi) {
        ///nope            auto& gs = gss[(yi * 3) + xi];
        ///nope
        ///nope            sum += gs * KIRSCH_MASKS_4[edge][xi][yi];
        ///nope        }
        ///nope    }
        ///nope
        ///nope    edgeTemp = sum;
        ///nope}

        { //

        

        // 1. get direction [0, 1, 2, 3] if all are same then it means just copy the pixel in all pixels after upscaling
        // 2. base on fx, fy we decide whether we add a pixel and weight or copy a pixel

            ///noper32 strongestEdge = max (
            ///nope    max (edgeTemps[0], edgeTemps[1]), 
            ///nope    max (edgeTemps[2], edgeTemps[3])
            ///nope);
            ///nope
            ///nopeu8 edgeDirection = 
            ///nope    (strongestEdge == edgeTemps[0]) ? 0 : 
            ///nope    (strongestEdge == edgeTemps[1]) ? 1 : 
            ///nope    (strongestEdge == edgeTemps[2]) ? 2 : 
            ///nope    3;


            //  ABOUT
            // Calculate the absolute difference between our main pixel and all 4 corners.
            //
            edgeTemps[0] = (gss[0] + gss[1] + gss[3]) / 3;
            edgeTemps[1] = (gss[1] + gss[2] + gss[5]) / 3;
            edgeTemps[2] = (gss[3] + gss[6] + gss[7]) / 3;
            edgeTemps[3] = (gss[5] + gss[7] + gss[8]) / 3;
            //
            edgeTemps[0] = abs (edgeTemps[0] - gss[4]);
            edgeTemps[1] = abs (edgeTemps[1] - gss[4]);
            edgeTemps[2] = abs (edgeTemps[2] - gss[4]);
            edgeTemps[3] = abs (edgeTemps[3] - gss[4]);


            // ??? How do i detect if a pixel is not connected to any of the corner pixels ?



            r32 blend = (
                gss[0] + gss[1] + gss[2] + gss[3] + 
                gss[5] + gss[6] + gss[7] + gss[8]) / 
                8.0f;

            // ??? TODO: what is that supposed to mean ???
            // todo i also need fills on horizontal and vertical

            //  ABOUT
            // If a new pixel is
            if (abs (blend - gss[4]) < 0.1f) {
                const uchar4& pixel = patch[4];

                #ifdef ATS_ENABLE_RENDER_TO_FILE
                    output[id + 0] = pixel.x;
                    output[id + 1] = pixel.y;
                    output[id + 2] = pixel.z;
                    output[id + 3] = 255;
                #else
                    surf2Dwrite<uchar4> (pixel, output, x * sizeof (uchar4), y);
                #endif

                return;
            }

            //  ABOUT
            // Strongest edge is where the central pixel matches the corner the most.
            //
            r32 strongestEdge = min (
                min (edgeTemps[0], edgeTemps[1]), 
                min (edgeTemps[2], edgeTemps[3])
            );

            u8 edgeDirection = 
                (strongestEdge == edgeTemps[0]) ? 0 : 
                (strongestEdge == edgeTemps[1]) ? 1 : 
                (strongestEdge == edgeTemps[2]) ? 2 : 
                3;

            // 0 -> LT, 1 -> TR, 2 -> BL, 3 -> RB
            u8 subpixelPos = 0 + (fx > 0.5) + ((fy > 0.5) * 2);

            if (edgeDirection == subpixelPos) { // --- copy
                auto& pixel = patch[(1 * 3) + 1]; // middle pixel
            
                #ifdef ATS_ENABLE_RENDER_TO_FILE
                    output[id + 0] = pixel.x;
                    output[id + 1] = pixel.y;
                    output[id + 2] = pixel.z;
                    output[id + 3] = 255;
                #else
                    surf2Dwrite<uchar4> (pixel, output, x * sizeof (uchar4), y);
                #endif
            } else {

                //const u8 corners [] { 0, 2, 6, 8 };
                //const u8 corners [] { 6, 8, 2, 0 };
                // patch[(0 * 3) + 0] -> 0 -> LT
                // patch[(0 * 3) + 2] -> 2 -> TR
                // patch[(2 * 3) + 0] -> 6 -> BL
                // patch[(2 * 3) + 2] -> 8 -> RB

                // TODO. This is wrong.
                //  corners 
                //uchar4 pixel = patch[corners[subpixelPos]];
                uchar4 pixel = { 0, 0, 0, 255};
            
                #ifdef ATS_ENABLE_RENDER_TO_FILE
                    output[id + 0] = pixel.x;
                    output[id + 1] = pixel.y;
                    output[id + 2] = pixel.z;
                    output[id + 3] = 255;
                #else
                    surf2Dwrite<uchar4> (pixel, output, x * sizeof (uchar4), y);
                #endif
            }

            //{
            //    uchar4 pixel { edgeTemps[0], edgeTemps[0], edgeTemps[0], 255};
            //
            //    #ifdef ATS_ENABLE_RENDER_TO_FILE
            //        output[id + 0] = pixel.x;
            //        output[id + 1] = pixel.y;
            //        output[id + 2] = pixel.z;
            //        output[id + 3] = 255;
            //    #else
            //        surf2Dwrite<uchar4> (pixel, output, x * sizeof (uchar4), y);
            //    #endif
            //}


        }

        ///noper32 gslt;
        ///noper32 gstr;
        ///noper32 gsrb;
        ///noper32 gsbl;
        ///noper32 gsmm;
        ///nope
        ///nope{
        ///nope    r32 u = ((((r32) ix) + (-1.0) + 0.5f) / iWidth);
        ///nope    r32 v = ((((r32) iy) + (-1.0) + 0.5f) / iHeight);
        ///nope
        ///nope    uchar4 pixel = tex2D<uchar4> (texture, u, v);
        ///nope    gslt = GRAYSCALE (pixel.x, pixel.y, pixel.z);
        ///nope}
        ///nope
        ///nope{
        ///nope    r32 u = ((((r32) ix) + (+1.0) + 0.5f) / iWidth);
        ///nope    r32 v = ((((r32) iy) + (-1.0) + 0.5f) / iHeight);
        ///nope
        ///nope    uchar4 pixel = tex2D<uchar4> (texture, u, v);
        ///nope    gstr = GRAYSCALE (pixel.x, pixel.y, pixel.z);
        ///nope}
        ///nope
        ///nope{
        ///nope    r32 u = ((((r32) ix) + (+1.0) + 0.5f) / iWidth);
        ///nope    r32 v = ((((r32) iy) + (+1.0) + 0.5f) / iHeight);
        ///nope
        ///nope    uchar4 pixel = tex2D<uchar4> (texture, u, v);
        ///nope    gsrb = GRAYSCALE (pixel.x, pixel.y, pixel.z);
        ///nope}
        ///nope
        ///nope{
        ///nope    r32 u = ((((r32) ix) + (-1.0) + 0.5f) / iWidth);
        ///nope    r32 v = ((((r32) iy) + (+1.0) + 0.5f) / iHeight);
        ///nope
        ///nope    uchar4 pixel = tex2D<uchar4> (texture, u, v);
        ///nope    gsbl = GRAYSCALE (pixel.x, pixel.y, pixel.z);
        ///nope}
        ///nope
        ///nope{
        ///nope    r32 u = ((((r32) ix) + (+0.0) + 0.5f) / iWidth);
        ///nope    r32 v = ((((r32) iy) + (+0.0) + 0.5f) / iHeight);
        ///nope
        ///nope    uchar4 pixel = tex2D<uchar4> (texture, u, v);
        ///nope    gsmm = GRAYSCALE (pixel.x, pixel.y, pixel.z);
        ///nope}
        ///nope
        ///nope// this is not a good edge detection
        ///nope//  but it does gives us pixels with edge only
        ///nope// however we need to get information
        ///nope//  is the edge lt,tr,bl,rb
        ///noper32 dirx = -((gslt + gstr) - (gsbl + gsrb));
        ///noper32 diry = +((gslt + gsbl) - (gstr + gsrb));

        ///nope{
        ///nope    // dirx > 128 -> edge, < 128 -> no edge
        ///nope}
        ///nope
        ///nope
        ///nope{ // --- output
        ///nope    u8 color = gss[0] * 255;
        ///nope
        ///nope    #ifdef ATS_ENABLE_RENDER_TO_FILE
        ///nope        output[id + 0] = color;
        ///nope        output[id + 1] = color;
        ///nope        output[id + 2] = color;
        ///nope        output[id + 3] = 255;
        ///nope    #else
        ///nope        uchar4 pixel { color, color, color, 255 };
        ///nope        surf2Dwrite<uchar4> (pixel, output, x * sizeof (uchar4), y);
        ///nope    #endif
        ///nope}

    }


    __global__ void EdgeB (
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

        // 0.0 -> ~1,0
        r32 px = (x + 0.5f) * ((r32) iWidth / oWidth); 
        r32 py = (y + 0.5f) * ((r32) iHeight / oHeight);

        auto ix = (u16) px;
        auto iy = (u16) py;

        r32 fx = px - ix;
        r32 fy = py - iy;

        uchar4 patch[9];
        r32 gss[9];

        for (u8 yi = 0; yi < 3; ++yi) {
            for (u8 xi = 0; xi < 3; ++xi) {

                r32 u = ((((r32) ix + xi) + (-1.0) + 0.5f) / iWidth);
                r32 v = ((((r32) iy + yi) + (-1.0) + 0.5f) / iHeight);

                auto& pixel = patch[(yi * 3) + xi];
                auto& gs = gss[(yi * 3) + xi]; 

                pixel = tex2D<uchar4> (texture, u, v);
                gs = GRAYSCALE (pixel.x, pixel.y, pixel.z);

            }
        }

        r32 edgeTemps[8];

        { //

            edgeTemps[0] = (gss[0] + gss[1] + gss[3]) / 3;
            edgeTemps[1] = (gss[1] + gss[2] + gss[5]) / 3;
            edgeTemps[2] = (gss[3] + gss[6] + gss[7]) / 3;
            edgeTemps[3] = (gss[5] + gss[7] + gss[8]) / 3;

            edgeTemps[3] = (gss[0] + gss[1] + gss[2]) / 3;
            edgeTemps[4] = (gss[2] + gss[5] + gss[8]) / 3;
            edgeTemps[5] = (gss[8] + gss[7] + gss[6]) / 3;
            edgeTemps[6] = (gss[6] + gss[3] + gss[0]) / 3;

            edgeTemps[0] = abs (edgeTemps[0] - gss[4]);
            edgeTemps[1] = abs (edgeTemps[1] - gss[4]);
            edgeTemps[2] = abs (edgeTemps[2] - gss[4]);
            edgeTemps[3] = abs (edgeTemps[3] - gss[4]);

            edgeTemps[4] = abs (edgeTemps[4] - gss[4]);
            edgeTemps[5] = abs (edgeTemps[5] - gss[4]);
            edgeTemps[6] = abs (edgeTemps[6] - gss[4]);
            edgeTemps[7] = abs (edgeTemps[7] - gss[4]);

            r32 blend = (
                gss[0] + gss[1] + gss[2] + gss[3] + 
                gss[5] + gss[6] + gss[7] + gss[8]) / 
                8.0f;

            if (abs (blend - gss[4]) < 0.1f) {
                const uchar4& pixel = patch[4];

                #ifdef ATS_ENABLE_RENDER_TO_FILE
                    output[id + 0] = pixel.x;
                    output[id + 1] = pixel.y;
                    output[id + 2] = pixel.z;
                    output[id + 3] = 255;
                #else
                    surf2Dwrite<uchar4> (pixel, output, x * sizeof (uchar4), y);
                #endif

                return;
            }

            // Strongest edge is where the central pixel matches the corner the most.
            //
            r32 strongestEdge = min (
                min (
                    min (edgeTemps[0], edgeTemps[1]), 
                    min (edgeTemps[2], edgeTemps[3])
                ),
                min (
                    min (edgeTemps[4], edgeTemps[5]), 
                    min (edgeTemps[6], edgeTemps[7])
                )
            );

            u8 edgeDirection = 
                (strongestEdge == edgeTemps[0]) ? 0 : 
                (strongestEdge == edgeTemps[1]) ? 1 : 
                (strongestEdge == edgeTemps[2]) ? 2 : 
                (strongestEdge == edgeTemps[3]) ? 3 : 
                (strongestEdge == edgeTemps[4]) ? 4 : 
                (strongestEdge == edgeTemps[5]) ? 5 : 
                (strongestEdge == edgeTemps[6]) ? 6 : 
                7;

            // 0 -> LT, 1 -> TR, 2 -> BL, 3 -> RB
            u8 subpixelPos = 0 + (fx > 0.5) + ((fy > 0.5) * 2);

            if (edgeDirection == subpixelPos) { // --- copy
                auto& pixel = patch[(1 * 3) + 1]; // middle pixel
            
                #ifdef ATS_ENABLE_RENDER_TO_FILE
                    output[id + 0] = pixel.x;
                    output[id + 1] = pixel.y;
                    output[id + 2] = pixel.z;
                    output[id + 3] = 255;
                #else
                    surf2Dwrite<uchar4> (pixel, output, x * sizeof (uchar4), y);
                #endif

                return;
            }

            //if ()
            
            {

                // LT, TR, BL, RB
                const u8 corners [] { 0, 2, 6, 8 };
                const uchar4& pixel = patch[corners[subpixelPos]];
                //const uchar4 pixel = { 0, 0, 0, 255};
            
                #ifdef ATS_ENABLE_RENDER_TO_FILE
                    output[id + 0] = pixel.x;
                    output[id + 1] = pixel.y;
                    output[id + 2] = pixel.z;
                    output[id + 3] = 255;
                #else
                    surf2Dwrite<uchar4> (pixel, output, x * sizeof (uchar4), y);
                #endif

            }
        }
    }

}
