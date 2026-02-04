// Made by Matthew Strumillo 2025.06.21
//
#include <blue/error.hpp>
//
#include <stb_image.h>
#include <stb_image_write.h>
//
#include <vector>


#define GRAYSCALE(r, g, b) \
    (((0.299f * r) + (0.587f * g) + (0.114f * b)))


void SSIM_MASKED ( // SSIM (gussian window + mask + luma-channels)
    r64& ssim,
    const u8* const& imageDataA,
    const u8* const& imageDataB,
    const u8* const& mask,
    const u32 & width,
    const u32 & height
) { 
    const r64 MAX = 1.0;
    const r64 C1 = (0.01 * MAX) * (0.01 * MAX);
    const r64 C2 = (0.03 * MAX) * (0.03 * MAX);

    std::vector<r32> Y_A(width * height);
    std::vector<r32> Y_B(width * height);
    std::vector<r32> M  (width * height);

    for (u32 i = 0; i < width * height; i++) {
        r32 rA = imageDataA[i*3+0] / 255.0f;
        r32 gA = imageDataA[i*3+1] / 255.0f;
        r32 bA = imageDataA[i*3+2] / 255.0f;

        r32 rB = imageDataB[i*3+0] / 255.0f;
        r32 gB = imageDataB[i*3+1] / 255.0f;
        r32 bB = imageDataB[i*3+2] / 255.0f;

        Y_A[i] = GRAYSCALE(rA, gA, bA);
        Y_B[i] = GRAYSCALE(rB, gB, bB);

        M[i] = mask ? (mask[i] / 255.0f) : 1.0f;
    }

    static constexpr int W = 11;
    static constexpr int R = 5;

    r32 G[W][W];
    r32 Gsum = 0.0f;

    for (int y = -R; y <= R; y++) {
        for (int x = -R; x <= R; x++) {
            r32 v = expf(-(x*x + y*y) / (2.0f * 1.5f * 1.5f));
            G[y+R][x+R] = v;
            Gsum += v;
        }
    }

    for (int i = 0; i < W; i++) {
        for (int j = 0; j < W; j++) {
            G[i][j] /= Gsum; 
        }
    }

    r64 ssim_sum = 0.0;
    r64 weight_sum = 0.0;

    for (u32 y = R; y < height - R; y++) {
        for (u32 x = R; x < width - R; x++) {

            r64 Wsum = 0.0;
            r64 Sx = 0.0, Sy = 0.0;
            r64 Sx2 = 0.0, Sy2 = 0.0, Sxy = 0.0;

            for (int j = -R; j <= R; j++) {
                for (int i = -R; i <= R; i++) {

                    u32 idx = (y + j) * width + (x + i);

                    r64 w = G[j+R][i+R] * M[idx];
                    if (w == 0.0) continue;

                    r64 a = Y_A[idx];
                    r64 b = Y_B[idx];

                    Wsum += w;
                    Sx   += w * a;
                    Sy   += w * b;
                    Sx2  += w * a * a;
                    Sy2  += w * b * b;
                    Sxy  += w * a * b;
                }
            }

            if (Wsum == 0.0)
                continue;

            r64 muA = Sx / Wsum;
            r64 muB = Sy / Wsum;

            r64 varA = Sx2 / Wsum - muA * muA;
            r64 varB = Sy2 / Wsum - muB * muB;
            r64 cov  = Sxy / Wsum - muA * muB;

            varA = fmaxf(varA, 0.0);
            varB = fmaxf(varB, 0.0);

            r64 num = (2*muA*muB + C1) * (2*cov + C2);
            r64 den = (muA*muA + muB*muB + C1) *
                      (varA + varB + C2);

            r64 ssim_local = num / den;

            ssim_sum += ssim_local * Wsum;
            weight_sum += Wsum;
        }
    }

    ssim = ssim_sum / weight_sum;
}


#define OUTPUT_ATS "res\\output.png"
#define OUTPUT_CYCLES "res\\render_0001.png"
#define OUTPUT_MASK "res\\mask.png"


s32 main (s32 argumentsCount, c8* arguments[]) {

    LOGINFO ("Compare build!\n");

    const c8 filePathA [] = OUTPUT_ATS;
    const c8 filePathB [] = OUTPUT_CYCLES;
    const c8 filePathC [] = OUTPUT_MASK;

    u64 byteLength;
    u8* imageDataA;
    u8* imageDataB;
    u8* imageDataC;

    u8 useChannels;
    s32 sChannels;
	s32 sWidth;
	s32 sHeight;

    { // Read file A
        auto& channels = sChannels;
		auto& width = sWidth;
		auto& height = sHeight;

		imageDataA = stbi_load (filePathA, &width, &height, &channels, 0);
		if (imageDataA == nullptr) ERROR ("Incorrect image filepath!\n");

        // Always only test RGB layers.
        LOGINFO ("channels: %d\n", sChannels);
        if (sChannels >= 3) { useChannels = 3; }
        else { LOGERROR ("Image has to have 3 channels!\n"); }
    }

    { // Read file B
        s32 channels;
		s32 width;
		s32 height;

		imageDataB = stbi_load (filePathB, &width, &height, &channels, 0);
		if (imageDataB == nullptr) ERROR ("Incorrect image filepath!\n");

        u8 isEqual = (sChannels == channels) && 
            (sWidth == width) &&
            (sHeight = height);

        if (!isEqual) {
            ERROR ("Images don't have the same channel, width, height size. Ensure images are equal in size!");
        }

        byteLength = width * height * channels;
    }

    { // Read file C
        s32 channels;
		s32 width;
		s32 height;

		imageDataC = stbi_load (filePathC, &width, &height, &channels, 0);
		if (imageDataC == nullptr) ERROR ("Incorrect image filepath!\n");
    }

    u32 pixelsCounter = 0; // pixels that differ counter.
    r64 average = 0.0f;
    u16 mean = 0.0f;
    r64 mse = 0;
    r64 rmse = 0;
    r64 psnr = 0;
    r64 gssim = 0;
    r64 ssim = 0;

    { // mean, median
        for (u64 i = 0; i < byteLength; ++i) {
            if (imageDataA[i] != imageDataB[i]) {
                ++pixelsCounter;

                u16 absolute = abs ((s16)imageDataA[i] - (s16)imageDataB[i]);
                average += absolute;

                if (absolute > mean) {
                    mean = absolute;
                }

            }
        }

        if (average != 0.0f) {
            average /= pixelsCounter;
        }
    }

    { // MSE
        for (u32 y = 0; y < sHeight; ++y) {
            for (u32 x = 0; x < sWidth; ++x) {
                for (u32 c = 0; c < useChannels; ++c) {
                    r32 a = (imageDataA[((y * sWidth) + x) * sChannels + c] / 255.0); // norm
                    r32 b = (imageDataB[((y * sWidth) + x) * sChannels + c] / 255.0); // norm
                    //mse += pow (fabs (a - b), 2);
                    r32 diff = a - b;
                    mse += diff * diff;
                }
            }
        }

        mse = mse / (sHeight * sWidth * useChannels);
    }

    { // RMSE
        rmse = sqrtf (mse);
    }


    { // PSNR // A difference of ~1e-5 in PSNR is absolutely expected.
        const r64 MAX = 1.0; // 255.0; // norm
        if (mse == 0) psnr = INFINITY; // identical
        psnr = 10.0 * log10((MAX * MAX) / mse);
    }

    {
        SSIM_MASKED (ssim, imageDataA, imageDataB, imageDataC, sWidth, sHeight);
    }

    stbi_image_free (imageDataA);
    stbi_image_free (imageDataB);
    stbi_image_free (imageDataC);

    //LOGINFO (
    //    "pixels: %i, median: %f, mean: %i\n",
    //    pixelsCounter, median, mean
    //);
    //LOGINFO ("mse: %f, psnr: %f, ssim: %f\n", mse, psnr, ssim);

    // RMSE is more readable - for example 0.33 translates to around 33% differ.

    r64 difPixelCoverage = (r64)(pixelsCounter) / byteLength;

    printf (
        "dif-pixels: %f, average: %f, mean: %i, \n"
        "mse: %f, rmse: %f, psnr: %f, ssim: %f, \n",
        difPixelCoverage, average, mean, mse, rmse, psnr, ssim
    );

	return 0;

}
