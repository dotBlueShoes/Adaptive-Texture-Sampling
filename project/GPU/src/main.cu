// Made by Matthew Strumillo 2024.07.20
//
#include "heading.cuh"
#include <blue/blue.hpp>
//
#include <glib/gl.hpp>
#include <glib/mesh.hpp>
#include <glib/shader.hpp>
#include <glib/uniform.hpp>
#include <glib/texture.hpp>
#include <glib/framebuffer.hpp>
#include <glib/camera.hpp>
#include <glib/transform.hpp>
//
#include "asset.hpp"
#include "image.hpp"
#include "uniform.hpp"
//
#include "cuda/framework.cuh"
#include "cuda/test.cuh"
#include "cuda/edge.cuh"
#include "cuda/edge2.cuh"
#include "texture.cuh"
//
#include <charconv>


void framebuffer_size_callback (
	GLFWwindow* window, 
	int width, 
	int height
) {
    //  IMPORTANT
    // unused variables -> width, height
	glViewport (0, 0, ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y);
}  


void GLFW_TERMINATE (void*) {
	glfwTerminate ();
}


void Filter (
    const cudaTextureObject_t&  texture,
    #ifdef ATS_ENABLE_RENDER_TO_FILE
        u8*&                    output,
    #else
        cudaSurfaceObject_t&    output,
    #endif
    const u16&                  oWidth,
    const u16&                  oHeight
) {

    { // --- FILTER
        const dim3 threads (16, 16);
        const dim3 blocks  (oHeight / 16, oWidth / 16);

        assert (THREADS_16 == threads.x);
        assert (THREADS_16 == threads.y);

        EDGE::TEXTURE::FILTERING::BicubicRW <<<blocks, threads>>> (
            texture, output, 320, 320, 640, 640
        );

        //EDGE2::EdgeA <<<blocks, threads>>> (
        //    texture, output, 320, 320, 640, 640
        //);

        //EDGE2::Nearest <<<blocks, threads>>> (
        //    texture, output, 320, 320, oWidth, oHeight
        //);

        //EDGE2::Linear <<<blocks, threads>>> (
        //    texture, output, 320, 320, 640, 640
        //);

    } // ---

    //  ABOUT
    // Why not use max threads? ->
    //  32 * 32 = 1024 -> which is max on my device.
    //  16 * 16 = 256
    //
    // Just because 1024 is my max it doesn't mean it can 
    //  actually be launched with that configuration.
    //
    // Running on max threads is ok if I don't hit a resource limit per block.
    //  If my kernel uses a lot of registers or shared memory per thread, CUDA 
    //  may not have enough resources to schedule the block.
    //

    //nope{ // dual line draw
    //nope    const dim3 threads (1);
    //nope    const dim3 blocks  (1);
    //nope
    //nope    const uchar3 colorA { 255, 255, 255 };
    //nope    const uchar3 colorB { 0, 0, 0 };
    //nope
    //nope    //const u32 start = (640 * 4 * 0) + (4 * 0);
    //nope    //const u32 end   = (640 * 4 * 10) + (4 * 20);
    //nope    u16 ps [2] {  0,  0 };
    //nope    u16 pe [2] { 20, 10 };
    //nope
    //nope    LINEDRAW::DLD <<<blocks, threads>>> (
    //nope        ps[0], ps[1], pe[0], pe[1], output, colorA, colorB
    //nope    );
    //nope}

    //{ // Output size TESTS
    //    const dim3 threads (16, 16);
    //    const dim3 blocks  (height / 16, width / 16);
    //
    //    assert (THREADS_16 == threads.x);
    //    assert (THREADS_16 == threads.y);
    //
    //    //EDGE::TEXTURE::FILTERING::ATS1 <<<blocks, threads>>> (
    //    //    texture, output, 320, 320, 640, 640
    //    //);
    //
    //    //EDGE::TEXTURE::FILTER::KirschAll <<<blocks, threads>>> (
    //    //    texture, output, 640, 640
    //    //);
    //
    //    //EDGE::TEXTURE::FILTER::KirschSingle <<<blocks, threads>>> (
    //    //    texture, output, 640, 640, EDGE::TEXTURE::FILTER::KIRSCH_DIRECTION_W
    //    //);
    //
    //    //EDGE::TEXTURE::FILTER::LaplacianA <<<blocks, threads>>> (
    //    //    texture, output, 640, 640
    //    //);
    //
    //    //EDGE::TEXTURE::FILTER::LaplacianB <<<blocks, threads>>> (
    //    //    texture, output, 320, 320, 640, 640
    //    //);
    //
    //    //EDGE::TEXTURE::FILTER::Sobel <<<blocks, threads>>> (
    //    //    texture, output, 640, 640
    //    //);
    //
    //    EDGE::TEXTURE::FILTERING::Bicubic <<<blocks, threads>>> (
    //        texture, output, 320, 320, 640, 640
    //    );
    //
    //    //EDGE::TEXTURE::FILTERING::CustomA <<<blocks, threads>>> (
    //    //    texture, output, 320, 320, 640, 640
    //    //);
    //
    //    //EDGE::TEXTURE::FILTERING::CustomB <<<blocks, threads>>> (
    //    //    texture, output, 320, 320, 640, 640
    //    //);
    //
    //}
    //
    //// { // Input size TESTS
    ////     const u16 iHeight = ATS_ASSET_INPUT_CANVAS_Y;
    ////     const u16 iWidth = ATS_ASSET_INPUT_CANVAS_X;
    //// 
    ////     const dim3 threads (32, 32);
    ////     const dim3 blocks  (iWidth / 32, iHeight / 32);
    //// 
    ////     EDGE::TEXTURE::FILTER::LaplacianA <<<blocks, threads>>> (
    ////         texture, output, iWidth, iHeight
    ////     );
    //// }
    //
    //
}


s32 main (s32 argumentsCount, c8* arguments[]) {

    // Reads from the framebuffer texture.
    cudaGraphicsResource* cudaInputTexture;

    #ifndef ATS_ENABLE_RENDER_TO_FILE
        //Writes into the framebuffer texture.
        cudaGraphicsResource* cudaOutputTexture;
    #endif

    // --- Top variables
    IMAGE::Head iImage;
    u8* iImageData;
    //
    IMAGE::Head specialImage;
    u8* specialImageData;
    //
    GLFWwindow* window;
    // ---


    // --- Bluelib Init.
    BINIT ("Starting Execution!\n");
    // ---

    DEBUG (DEBUG_FLAG_LOGGING) { CUDA::IsAsynchronousKernelLaunches (); }

    // --- Args interpreter  
        c8* imageFilepath;
        c8* specialFilepath;
        //u8 imageChannels;

        switch (argumentsCount) {

            //  ISSUE. HARDCODED and No validation.
            //

            case 3: {
                const u8 iFilepath = 1;
                const u8 sFilepath = 2;

                LOGINFO ("[%d]-arg: %s\n", iFilepath - 1, arguments[iFilepath]);
                LOGINFO ("[%d]-arg: %s\n", sFilepath - 1, arguments[sFilepath]);

                imageFilepath = arguments[iFilepath];
                specialFilepath = arguments[sFilepath];
                //imageChannels = 3;
            } break;

            case 4: {
                const u8 iFilepath = 1;
                const u8 sFilepath = 2;
                const u8 iChannels = 3;

                LOGINFO ("[%d]-arg: %s\n", iFilepath - 1, arguments[iFilepath]);
                LOGINFO ("[%d]-arg: %s\n", sFilepath - 1, arguments[sFilepath]);
                LOGINFO ("[%d]-arg: %s\n", iChannels - 1, arguments[iChannels]);

                imageFilepath = arguments[iFilepath];
                specialFilepath = arguments[sFilepath];
                //imageChannels = atoi(arguments[iChannels]);
            } break;

            default:
                LOGWARN ("Invalid number of arguments passed!\n");
            case 1: { // Apply DEFAULT values for arguments.
                imageFilepath   = (c8*)(void*)ASSET::FILE_I0_NOALPHA;
                specialFilepath = (c8*)(void*)ASSET::FILE_I0_UPSCALE;
                //imageChannels = 3;
            };

        }
    // ---
	
    // LOGINFO ("imageFilepath: %s\n", imageFilepath);

    // --- Persistent asset loading.
	IMAGE::Load (iImage, iImageData, imageFilepath);
    IMAGE::Load (specialImage, specialImageData, specialFilepath);
    // ---


    { // Ensure 32 base on images.
        if (iImage.width % 32 != 0 || iImage.height % 32 != 0)
            ERROR ("CUDA kernels operate on 32x base. Supply an Image with a base of 32px.\n");
    }

    #ifdef ATS_DISPLAY_FPS
        c8 windowTitleBuffer[] = ATS_ASSET_WINDOW_TITLE " \0\0\0\0\0\0\0\0\0";
    #endif

	{ // --- Window, API's setup's

		glfwInit ();
		glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 6);
		glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        #ifndef ATS_ENABLE_FRAMEBUFFER_RENDER

            // Make the default framebuffer able to support MSAA
            //  If we're using MSAA and declare a custom framebuffer for rendering then 
            //  this step is not necessary.
            //
            #if ATS_MSAA_LEVEL != 0
                glfwWindowHint (GLFW_SAMPLES, ATS_MSAA_LEVEL);
            #endif

        #endif

        // Create Window.
        #ifdef ATS_DISPLAY_FPS
            window = glfwCreateWindow (
			    ATS_ASSET_OUTPUT_CANVAS_X, ATS_ASSET_OUTPUT_CANVAS_Y, 
			    windowTitleBuffer, 
			    nullptr, nullptr
		    );
        #else
            window = glfwCreateWindow (
			    ATS_ASSET_OUTPUT_CANVAS_X, ATS_ASSET_OUTPUT_CANVAS_Y, 
			    ATS_ASSET_WINDOW_TITLE, 
			    nullptr, nullptr
		    );
        #endif

		{ // --- Properly deallocate data if we hit ERROR.
			// TODO MEMORY::EXIT::PUSH (IMAGE::Free, iImage.data);
			MEMORY::EXIT::PUSH (GLFW_TERMINATE, nullptr);
		}

		if (window == nullptr) ERROR ("Failed to create GLFW window!\n");

		// --- Bind GL context to window.
		glfwMakeContextCurrent (window);

        //  NOTE
        // In the context of 320 x 320 images actually not needed.
		// More GLFW initialization.
		// glfwSetFramebufferSizeCallback (window, framebuffer_size_callback);

        #ifdef ATS_SWAP_INTERVAL
            glfwSwapInterval (ATS_SWAP_INTERVAL);
        #endif

		// --- Initialize GLAD.
		if (!gladLoadGLLoader ((GLADloadproc)glfwGetProcAddress)) {
			ERROR ("Failed to initialize GLAD!\n");
		}

        #ifdef ATS_ENABLE_EXTENSIONS_CHECK

            LOGINFO ("GL device: %s\n", glGetString(GL_VENDOR));
            GL::FunctionalityCheck ();
            
            //  TODO
            // - Pack it nicely into CUDA namespace functions.
            #ifdef ATS_ENABLE_DEEP_DEBUG
            { 
                s32 runtimeVersion = 0, driverVersion = 0;
                u32 glCudaDevicesCount;

                cudaGLGetDevices (&glCudaDevicesCount, nullptr, 0, cudaGLDeviceList::cudaGLDeviceListAll);
                LOGINFO ("CUDA devices supporting GL interop: %d\n", glCudaDevicesCount);

                cudaRuntimeGetVersion (&runtimeVersion);
                cudaDriverGetVersion (&driverVersion);

                LOGINFO ("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
                LOGINFO ("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 1000) / 10);
            }
            #endif

        #endif

        { // Ensure CUDA uses the same device that OpenGL uses.
            cudaDeviceProp deviceProperties;
            s32 device;

            cudaError_t errorCuda = cudaSetDevice (0);
            CUDA_GET_ERROR (errorCuda, "cudaSetDevice");

            #ifdef ATS_ENABLE_CUDA_DEVICE_CHECK
            { // Read CUDA device caps.
                cudaGetDevice           (&device);
                cudaGetDeviceProperties (&deviceProperties, device);
                DEBUG (DEBUG_FLAG_LOGGING) { CUDA::LogProperties (deviceProperties); }
            }
            #endif
        }

    } // ---


    /// { // --- Quick TESTs
    ///     TEST::FaultyKernel ();
    /// }


    { // --- Render

        // --- Meta-objects

		    //  LOGIC
		    // Defing an Object as Mesh, Material, Uniforms, Textures as if read from a file ( but not really ).

		    MESH::Mesh meshes[] {
		    	MESH::Mesh { 0, 0, MESH::DDD::CUBE::ELEMENTS_COUNT, MESH::DDD::CUBE::MODE },
		    	MESH::Mesh { 1, 0, MESH::DDD::FSQUARE::ELEMENTS_COUNT, MESH::DDD::FSQUARE::MODE },
		    };

		    CAMERA::Camera3D camera { 
		    	CAMERA::Camera3D::Type::TYPE_FREE, 
		    	glm::vec3 (.0f, .0f, ATS_CAMERA_Z_SIGN 3.0f), 
		    	ATS_CAMERA_FOV, 
		    	glm::vec3 (.0f, .0f, .0f)
		    };

            TRANSFORM::Local transforms[] {
                TRANSFORM::Local { 
                    ATS_TRANSFORM_POSITION_X, ATS_TRANSFORM_POSITION_Y, ATS_TRANSFORM_POSITION_Z, 
                    ATS_TRANSFORM_ROTATION_X, ATS_TRANSFORM_ROTATION_Y, ATS_TRANSFORM_ROTATION_Z,
                    ATS_TRANSFORM_SCALE_X,    ATS_TRANSFORM_SCALE_Y,    ATS_TRANSFORM_SCALE_Z
                },
            };

		    //LOGINFO (
		    //	"t: %d, p: [%f, %f, %f], v: %f, r: [%f, %f, %f]\n", 
		    //	camera.type, camera.position.x, 
		    //	camera.position.y, camera.position.z,
		    //	camera.fov, camera.rotation.x,
		    //	camera.rotation.y, camera.rotation.z
		    //);

        // ---


        // --- Render scene and environment variables.

            // Meshes
		    u32 vaosCount = 2;              
		    GLuint* vaos;					

            // Mesh buffers ( We have 2 VUE Meshes )
		    u32 xbosCount = MESH::VAO::VUE_BUFFERS_COUNT * 2;   
		    GLuint* xbos;

            const u32 XBOS_1_BUFFERS = 0;
            const u32 XBOS_2_BUFFERS = 3;

            // Material->Shader 's
		    u32 shadersCount = 2;           
		    GLuint* shaders;				

            // Material->Shader->Uniforms 's
            u32 uniformsListsCount = 8;     
		    u16* uniformsLists;             

            // Material->Texture 's
		    u32 texturesCount = 1;          
		    GLuint* textures;	

            #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER

                // Framebuffer 's
                u32 fbosCount = 1;               
		        GLuint* fbos;

                // Framebuffer->Texture 's
                u32 ftosCount = 1;          
		        GLuint* ftos;	

                // Framebuffer->RenderBuffer 's
                u32 rbosCount = 1;               
		        GLuint* rbos;

            #endif

            GLuint cudaglInteropTexture;

        // ---


		// --- Rendererer scene and environment setting up.

            { // --- Settings

                { // --- Activate Texture Handlers
                    glActiveTexture (GL_TEXTURE0);
                }

                #if ATS_ANISOTROPY_LEVEL != 0
                    GLfloat anisotropyMax = 0.0f;
                    glGetFloatv (GL_MAX_TEXTURE_MAX_ANISOTROPY, &anisotropyMax);

                    if (ATS_ANISOTROPY_LEVEL > anisotropyMax) 
                        ERROR ("Anisotropy Max is: %f\n", anisotropyMax);
                #endif

                #if ATS_MSAA_LEVEL != 0

                    // Custom framebuffer works independently from this setting.
                    //  Only the default framebuffer requires this call. 
                    //
                    #ifndef ATS_ENABLE_FRAMEBUFFER_RENDER
                        glEnable (GL_MULTISAMPLE);
                    #endif

                    // Check whats the MAXIMUM level supported by the device.
                    //
                    DEBUG (DEBUG_FLAG_LOGGING) {
                        GLint maxSamples;
                        glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);

                        LOGINFO ("MSAA max-samples-level: %d\n", maxSamples);

                        if (ATS_MSAA_LEVEL > maxSamples) {
                            ERROR ("ATS_MSAA_LEVEL is set higher then possible: %d\n", ATS_MSAA_LEVEL);
                        }
                    }

                #endif

            } // ---


			{ // --- Load Textures.
				ALLOCATE (GLuint, textures, texturesCount * sizeof (GLuint));
				MEMORY::EXIT::PUSH (FREE, textures);

				glGenTextures (texturesCount, textures);

                #if ATS_TEXTURE_FILTERING_METHOD == ATS_NEAREST
                
                    #if ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_NONE
                
				        TEXTURE::Create (textures[0], iImage, iImageData, GL_NEAREST, GL_NEAREST);
                
                    #elif ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_NEAREST
                
                        TEXTURE::CreateMipmapped (textures[0], iImage, iImageData, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST);
                
                    #elif ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_LINEAR
                
                        TEXTURE::CreateMipmapped (textures[0], iImage, iImageData, GL_LINEAR_MIPMAP_NEAREST, GL_NEAREST);
                
                    #endif
                
                #elif ATS_TEXTURE_FILTERING_METHOD == ATS_LINEAR
                
                    #if ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_NONE
                
				        TEXTURE::Create (textures[0], iImage, iImageData, GL_NEAREST, GL_LINEAR);
                
                    #elif ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_NEAREST
                
                        TEXTURE::CreateMipmapped (textures[0], iImage, iImageData, GL_NEAREST_MIPMAP_NEAREST, GL_LINEAR);
                
                    #elif ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_LINEAR
                        
                        TEXTURE::CreateMipmapped (textures[0], iImage, iImageData, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR);
                
                    #endif

                #elif ATS_TEXTURE_FILTERING_METHOD == ATS_EDGE 

                    // --- NOTE
                    // This is only for testing purposes.
                    //
                    TEXTURE::Create (textures[0], iImage.width, iImage.height);
                
                #endif
            } // ---


            { // --- Release CPU Textures
                // ISSUE. This asset is being deallocated between allocations
                //  To ensure Error-hit memory release a different approach is needed.
                // SOLUTION. loadBuffer -> allocated at the beginning with a specified max-size 
                // that should be able to load both textures and shader's text. So the allocation
                // for shaders cpu buffers are not needed.
                // Opt. 'stbi_image_free()' uses a simple free() call I could instead 
                //  allocate more than one image and release them with one call too.
                IMAGE::Free (iImageData); // TODO MEMORY::EXIT::POP ();
                IMAGE::Free (specialImageData); // TODO MEMORY::EXIT::POP ();
            }


            #ifndef ATS_ENABLE_RENDER_TO_FILE
            {
                glGenTextures (1, &cudaglInteropTexture);

                FRAMEBUFFER::TEXTURE::Create (
                    cudaglInteropTexture, GL_RGBA8, GL_RGBA,
                    ATS_ASSET_OUTPUT_CANVAS_X, ATS_ASSET_OUTPUT_CANVAS_Y, 
                    GL_NEAREST, GL_UNSIGNED_BYTE
                );
            }
            #endif


            #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER

                { // --- Create Framebuffer Textures.
                    ALLOCATE (GLuint, ftos, ftosCount * sizeof (GLuint));
			    	MEMORY::EXIT::PUSH (FREE, ftos);

                    glGenTextures (ftosCount, ftos);

                    #if ATS_FRAMEBUFFER_FILTERING_METHOD == ATS_NEAREST

                        #if ATS_MSAA_LEVEL == 0
			    	        FRAMEBUFFER::TEXTURE::Create (
                                ftos[0], GL_RGB8, GL_RGB,
                                ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y, 
                                GL_NEAREST, GL_UNSIGNED_BYTE
                            );
                        #else
                            FRAMEBUFFER::MSAA::TEXTURE::Create (
                                ftos[0], GL_RGB8, GL_RGB,
                                ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y
                            );
                        #endif

                    #elif ATS_FRAMEBUFFER_FILTERING_METHOD == ATS_LINEAR

                        #if ATS_MSAA_LEVEL == 0
                            FRAMEBUFFER::TEXTURE::Create (
                                ftos[0], GL_RGB8, GL_RGB,
                                ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y, 
                                GL_LINEAR, GL_UNSIGNED_BYTE
                            );
                        #else
                            FRAMEBUFFER::MSAA::TEXTURE::Create (
                                ftos[0], GL_RGB8, GL_RGB,
                                ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y
                            );
                        #endif

                    #endif

			    } // ---


			    { // --- Create Render Buffer Object -> Stencil & Depth
                    ALLOCATE (GLuint, rbos, rbosCount * sizeof (GLuint));
			    	MEMORY::EXIT::PUSH (FREE, rbos);

			    	glGenRenderbuffers (rbosCount, rbos);

                    #if ATS_MSAA_LEVEL == 0
			    	    FRAMEBUFFER::RENDERBUFFEROBJECT::Create (
                            rbos[0], GL_DEPTH24_STENCIL8, 
                            ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y
                        );
                    #else
                        FRAMEBUFFER::MSAA::RENDERBUFFEROBJECT::Create (
                            rbos[0], GL_DEPTH24_STENCIL8, 
                            ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y
                        );
                    #endif
			    } // ---


			    { // --- Create Framebuffer
                    ALLOCATE (GLuint, fbos, fbosCount * sizeof (GLuint));
			    	MEMORY::EXIT::PUSH (FREE, fbos);

			    	glGenFramebuffers (fbosCount, fbos);

                    #if ATS_MSAA_LEVEL == 0
			    	    FRAMEBUFFER::Create (
                            fbos[0], ftos[0], GL_COLOR_ATTACHMENT0, 
                            rbos[0], GL_DEPTH_STENCIL_ATTACHMENT
                        ); 
                    #else
                        FRAMEBUFFER::MSAA::Create (
                            fbos[0], ftos[0], GL_COLOR_ATTACHMENT0, 
                            rbos[0], GL_DEPTH_STENCIL_ATTACHMENT
                        ); 
                    #endif
			    } // ---


                // --- CUDA - Interpolation way. Requires a READ -> FRAMEBUFFER and a WRITE -> SWAP_TEXTURE.
                    cudaError_t errorCuda;

                    #ifndef ATS_ENABLE_RENDER_TO_FILE
                        errorCuda = cudaGraphicsGLRegisterImage (
                            &cudaOutputTexture, cudaglInteropTexture, GL_TEXTURE_2D, 
                            // maybe not needed
                            //cudaGraphicsRegisterFlagsSurfaceLoadStore |
                            cudaGraphicsRegisterFlagsWriteDiscard
                        );

                        CUDA_GET_ERROR (errorCuda, "cudaGraphicsGLRegisterImage-ftos-output");
                    #endif
                    
                    errorCuda = cudaGraphicsGLRegisterImage (
                        &cudaInputTexture, ftos[0], GL_TEXTURE_2D, 
                        cudaGraphicsRegisterFlagsReadOnly
                    );

                    CUDA_GET_ERROR (errorCuda, "cudaGraphicsGLRegisterImage-ftos-input");
                    

                    cudaResourceDesc rDescriptor = {};
                    rDescriptor.resType = cudaResourceTypeArray;

                    cudaTextureDesc tDescriptor     = {};
                    tDescriptor.addressMode[0]      = cudaAddressModeClamp;     // clamp texture x
                    tDescriptor.addressMode[1]      = cudaAddressModeClamp;     // clamp texture y
                    tDescriptor.filterMode          = cudaFilterModePoint;      // nearest-filtering
                    tDescriptor.readMode            = cudaReadModeElementType;  // use values as in opengl
                    tDescriptor.normalizedCoords    = 1;                        // use normalized coords
                // ---

            #endif

			{ // --- Load Shaders.
				
				ALLOCATE (GLuint, shaders, shadersCount * sizeof (GLuint));
				MEMORY::EXIT::PUSH (FREE, shaders);

                //  LOGIC
                // A buffer is required to read file. What's read is then compiled.

				const u16 bufferSize = 
                    ATS_MAX_CHARS_FOR_SHADER_VERT_FILE +
                    ATS_MAX_CHARS_FOR_SHADER_FRAG_FILE;

				c8* buffer;

				ALLOCATE (c8, buffer, bufferSize * sizeof (c8));
				MEMORY::EXIT::PUSH (FREE, buffer);

                c8* bufferFrag = buffer + ATS_MAX_CHARS_FOR_SHADER_VERT_FILE;
                c8* bufferVert = buffer;

                #ifdef ATS_GLSL_INJECTION

                    // TODO. Could optimize it so said allocation would only happen once and not for each shader.
                    u32 ffsdSize = ATS_MAX_CHARS_FOR_SHADER_FRAG_FILE + GLSL_INJECTION_INCLUDE_ALL_LENGTH; 

                    ALLOCATE (c8, GLSL::ffsd, ffsdSize * sizeof (c8));
			        MEMORY::EXIT::PUSH (FREE, GLSL::ffsd);

                #endif

				{ // Actual shader loading.

                    { // - shader 0
                        SHADER::ReadFile (ASSET::SHADER_VERT_CUBE, bufferVert, ATS_SHADER_VERT);
                        SHADER::ReadFile (ASSET::SHADER_FRAG_CUBE, bufferFrag, ATS_SHADER_FRAG);

                        SHADER::Compile (shaders[0], bufferVert, bufferFrag);
                    }

                    { // - shader 1
                        SHADER::ReadFile (ASSET::SHADER_VERT_FRAMEBUFFER, bufferVert, ATS_SHADER_VERT);
                        SHADER::ReadFile (ASSET::SHADER_FRAG_FRAMEBUFFER, bufferFrag, ATS_SHADER_FRAG);

                        SHADER::Compile (shaders[1], bufferVert, bufferFrag);
                    }
					
				}

                #ifdef ATS_GLSL_INJECTION

                    FREE (GLSL::ffsd); MEMORY::EXIT::POP ();

                #endif

				FREE (buffer); MEMORY::EXIT::POP ();
			} // ---


			// --- Set Shaders uniforms
				ALLOCATE (u16, uniformsLists, uniformsListsCount * sizeof (u16));
				MEMORY::EXIT::PUSH (FREE, uniformsLists);

				//  IMPORTANT
				// Remember to manually change the SHADER_N_UNIFORMS variables when changing shaders code!
				//
                const u32 SHADER_FINAL_UNIFORMS = 0;
                const u32 SHADER_SCENE_UNIFORMS = 2;

				uniformsLists[0] = 1;

				#if ATS_MSAA_LEVEL == 0
                    uniformsLists[1] = UNIFORM::ID::TEXTURE0;
                #else
                    uniformsLists[1] = UNIFORM::ID::MULTISAMPLE_TEXTURE;
                #endif

				uniformsLists[2] = 5;
				uniformsLists[3] = UNIFORM::ID::COLOR;
				uniformsLists[4] = UNIFORM::ID::TEXTURE0;
				uniformsLists[5] = UNIFORM::ID::PROJECTION;
				uniformsLists[6] = UNIFORM::ID::VIEW;
				uniformsLists[7] = UNIFORM::ID::TRANSFORM;
			// ---


			{ // --- Generate Meshes ( as in MEMORY space )

				//  ABOUT
				// To easily allocate and deallocate all defined meshes
				//  we're taking care of all buffers of said meshes 
				//  once.
				//

				//  IMPORTANT NOTE
				// Remember to update this value whenever a different
				//  vao is being used.
				//
				ALLOCATE (GLuint, xbos, xbosCount * sizeof (GLuint)); 
				MEMORY::EXIT::PUSH (FREE, xbos);

				glGenBuffers (xbosCount, xbos);

			} // ---


			{ // --- Initialize Meshes.

				ALLOCATE (GLuint, vaos, vaosCount * sizeof (GLuint)); 
				MEMORY::EXIT::PUSH (FREE, vaos);

				glGenVertexArrays (vaosCount, vaos);

                // Cube
				MESH::VAO::CreateVUE (
					vaos[0], xbos + XBOS_1_BUFFERS,
					MESH::DDD::CUBE::POINTS_COUNT,
					MESH::DDD::CUBE::VERTICES,
					MESH::DDD::CUBE::UVS,
					MESH::DDD::CUBE::ELEMENTS_COUNT,
					MESH::DDD::CUBE::ELEMENTS
				);

                // Screen
				MESH::VAO::CreateVUE (
					vaos[1], xbos + XBOS_2_BUFFERS,
					MESH::DDD::FSQUARE::POINTS_COUNT,
					MESH::DDD::FSQUARE::VERTICES,
					MESH::DDD::FSQUARE::UVS,
					MESH::DDD::FSQUARE::ELEMENTS_COUNT,
					MESH::DDD::FSQUARE::ELEMENTS
				);

			} // ---

		// ---


		{ // --- CANVAS SETTING UP

		    const r32 canvasRatio = (r32)ATS_ASSET_INPUT_CANVAS_X / (r32)ATS_ASSET_INPUT_CANVAS_Y;
		    glViewport (0, 0, ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y);
        
		    { // --- UNIFORMS PRE-SETUP

		    	{ // PROJECTION
		    		auto&& projection = *(glm::mat4*)(UNIFORM::projection + 1);
		    		auto&& count = *(GLint*)UNIFORM::projection;

		    		projection = GLM::VIEW::Perspective (camera.fov, canvasRatio);

		    		count = 1;
		    	}

		    	{ // VIEW
		    		auto&& view = *(glm::mat4*)(UNIFORM::view + 1);
		    		auto&& count = *(GLint*)UNIFORM::view;

		    		CAMERA::GetView (view, camera);

		    		count = 1;
		    	}

		    	{ // TRANSFORM
		    		//auto&& transform = *(glm::mat4*)(UNIFORM::transform + 1);
		    		auto&& count = *(GLint*)UNIFORM::transform;

		    		// NOTE. We're presetting TRANSFORM to identity matrix later.
		    		count = 1;
		    	}

		    } // ---

        } // ---


        // --- PREPERE CUDA SWAPPING SPACE
            const IMAGE::Head oImage { 4, ATS_ASSET_OUTPUT_CANVAS_Y, ATS_ASSET_OUTPUT_CANVAS_X };

            // Byte length of the final image.
            const u64 pixelsSize = oImage.width * oImage.height * oImage.channels;

            u8* pixels; ALLOCATE (u8, pixels, pixelsSize);
            MEMORY::EXIT::PUSH (FREE, pixels);

            #ifdef ATS_ENABLE_RENDER_TO_FILE

                //const GLenum outputChannelsType = GL_RGBA;
                #if defined(ATS_ENABLE_FRAMEBUFFER_RENDER) && ATS_OUTPUT_FILTERING_METHOD == ATS_EDGE
                    u8* output; cudaMalloc (&output, pixelsSize);
                #endif

            #endif

        // ---


        LOGINFO ("main-render-loop-enter\n");

        // --- frame time calculation
        r32 prevTimeValue = glfwGetTime ();
        // --- ensure the initial calculation is positive.
        r32 timeValue = prevTimeValue + FLT_EPSILON;


        // --- The game loop
		while (!glfwWindowShouldClose (window)) { 

			r32 deltaTime = timeValue - prevTimeValue;

            #ifdef ATS_DISPLAY_FPS
                const auto&& buffer = windowTitleBuffer + 11;
                const auto fps = 1 / deltaTime;
                std::to_chars (buffer, buffer + 9, fps, std::chars_format::fixed, 2);
                windowTitleBuffer [11 + 7] = '\0';
                glfwSetWindowTitle (window, windowTitleBuffer);
            #endif

			{ // --- My framebuffer or default render.

				#ifdef ATS_ENABLE_FRAMEBUFFER_RENDER
				    glBindFramebuffer (GL_FRAMEBUFFER, fbos[0]);
                #else
                    glBindFramebuffer (GL_FRAMEBUFFER, 0);
                #endif

				glEnable (GL_DEPTH_TEST); 
				glClearColor (ATS_ASSET_BACKGROUND_COLOR);
				glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				SHADER::Use (shaders[0]);

				{ // TEXTURE
					UNIFORM::texture0[0] = 0;
					UNIFORM::texture0[1] = textures[0];
				}

				{ // COLOR
					// TODO. MAGIC-NUMBERS. Changing the GREEN value.
					UNIFORM::color[1] = (sin (timeValue) / 2.0f) + 0.5f;
				}

				{ // Single mesh draw. 

					{ // TRANSFORM
						auto&& transform = *(glm::mat4*)(UNIFORM::transform + 1);
                        auto& local = transforms[0];

						transform = glm::mat4 (1.0f);

						transform = glm::translate (transform, 
                            glm::vec3 (local.px, local.py, local.pz)
                        ); 

                        #ifdef ATS_ROTATE_OBJECT

                            local.rx += 2 * deltaTime;
                            local.ry += 3 * deltaTime;
                            local.rz += 1 * deltaTime;

                        #endif

                        GLM::RotateByDegrees (
                            transform, 
                            local.rx, 
                            local.ry, 
                            local.rz
                        );

					}

					auto&& uniformsList = uniformsLists + SHADER_SCENE_UNIFORMS;
					SHADER::Set (UNIFORM::uniforms, uniformsList);

					const auto mesh = meshes[0];
        			glBindVertexArray (vaos[mesh.vaoIndex]);
					MESH::draws[mesh.drawIndex](mesh.elementsCount, mesh.drawMode);
        			glBindVertexArray (0);

				}

			} // ---

            #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER

                cudaError_t errorCuda;

                errorCuda = cudaGraphicsMapResources (1, &cudaInputTexture, 0);
                CUDA_GET_ERROR (errorCuda, "cudaGraphicsMapResources-ftos-input");

                #ifndef ATS_ENABLE_RENDER_TO_FILE
                    errorCuda = cudaGraphicsMapResources (1, &cudaOutputTexture, 0);
                    CUDA_GET_ERROR (errorCuda, "cudaGraphicsMapResources-ftos-output");

                    cudaArray_t outputTextureData;

                    cudaGraphicsSubResourceGetMappedArray (&outputTextureData, cudaOutputTexture, 0, 0);
                    CUDA_GET_ERROR (errorCuda, "cudaGraphicsSubResourceGetMappedArray-ftos-output");

                    cudaResourceDesc sDescriptor { };
                    sDescriptor.resType = cudaResourceTypeArray;
                    sDescriptor.res.array.array = outputTextureData;

                    cudaSurfaceObject_t output; // surface

                    cudaCreateSurfaceObject (&output, &sDescriptor);
                    CUDA_GET_ERROR (errorCuda, "cudaCreateSurfaceObject-ftos");
                #endif

                cudaArray* inputTextureData;

                errorCuda = cudaGraphicsSubResourceGetMappedArray (&inputTextureData, cudaInputTexture, 0, 0);
                CUDA_GET_ERROR (errorCuda, "cudaGraphicsSubResourceGetMappedArray-ftos-input");

                rDescriptor.res.array.array = inputTextureData;
                cudaTextureObject_t texture;

                errorCuda = cudaCreateTextureObject (&texture, &rDescriptor, &tDescriptor, nullptr);
                CUDA_GET_ERROR (errorCuda, "cudaCreateTextureObject-ftos");

                Filter (texture, output, oImage.width, oImage.height);

                //{ // --- FILTER
                //    const dim3 threads (16, 16);
                //    const dim3 blocks  (oImage.height / 16, oImage.width / 16);
                //
                //    assert (THREADS_16 == threads.x);
                //    assert (THREADS_16 == threads.y);
                //
                //    EDGE::TEXTURE::FILTERING::BicubicRW <<<blocks, threads>>> (
                //        texture, output, 320, 320, 640, 640
                //    );
                //
                //    //EDGE::TEXTURE::FILTERING::BicubicRW <<<blocks, threads>>> (
                //    //    texture, output, 320, 320, 640, 640
                //    //);
                //} // ---

                errorCuda = cudaDestroyTextureObject (texture);
                CUDA_GET_ERROR (errorCuda, "cudaDestroyTextureObject-ftos");

                errorCuda = cudaGraphicsUnmapResources (1, &cudaInputTexture, 0);
                CUDA_GET_ERROR (errorCuda, "cudaGraphicsUnmapResources-ftos-input");

                #ifndef ATS_ENABLE_RENDER_TO_FILE
                    errorCuda =  cudaDestroySurfaceObject (output);
                    CUDA_GET_ERROR (errorCuda, "cudaDestroySurfaceObject-ftos");

                    errorCuda = cudaGraphicsUnmapResources (1, &cudaOutputTexture, 0);
                    CUDA_GET_ERROR (errorCuda, "cudaGraphicsUnmapResources-ftos-output");
                #endif

            #endif

            #ifdef ATS_ENABLE_RENDER_TO_FILE
                break; 
            #endif

            #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER
                { // --- Default framebuffer render.

                    glViewport (0, 0, ATS_ASSET_OUTPUT_CANVAS_X, ATS_ASSET_OUTPUT_CANVAS_Y);

			    	glBindFramebuffer (GL_FRAMEBUFFER, 0);

    		    	glDisable (GL_DEPTH_TEST);
			    	glClear (GL_COLOR_BUFFER_BIT);

			    	SHADER::Use (shaders[1]);

			    	{ // TEXTURE
			    		UNIFORM::texture0[0] = 0;
			    		//UNIFORM::texture0[1] = ftos[0];
                        UNIFORM::texture0[1] = cudaglInteropTexture;
			    	}

                    //#ifdef ATS_ENABLE_FRAMEBUFFER_DEPTH_TEXTURE
                    //
                    //    { // DEPTH TEXTURE
			    	//    	UNIFORM::texture1[0] = 1;
			    	//    	UNIFORM::texture1[1] = ftos[1];
			    	//    }
                    //    
                    //#endif

			    	{ // DRAW Framebuffer contents

			    		auto&& uniformsList = uniformsLists + SHADER_FINAL_UNIFORMS;
			    		SHADER::Set (UNIFORM::uniforms, uniformsList);

			    		const auto mesh = meshes[1];
			    		glBindVertexArray (vaos[mesh.vaoIndex]);
			    		MESH::draws[mesh.drawIndex](mesh.elementsCount, mesh.drawMode);
        	    		glBindVertexArray (0);

			    	}

			    } // ---
            #endif

			// --- Apply and prepere for next.
			    glfwSwapBuffers (window);
			    glfwPollEvents ();

                glViewport (0, 0, ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y);
            // ---

            prevTimeValue = timeValue;
            timeValue = glfwGetTime ();

		} // ---


        // OLD
        // CUDA::Render (finalTexture);


        // --- Cleanup - CUDA
            
            #if defined(ATS_ENABLE_FRAMEBUFFER_RENDER) && ATS_OUTPUT_FILTERING_METHOD == ATS_EDGE

                #ifndef ATS_ENABLE_RENDER_TO_FILE
                    
                    errorCuda = cudaGraphicsUnregisterResource (cudaOutputTexture);
                    CUDA_GET_ERROR (errorCuda, "cudaGraphicsUnregisterResource-ftos-output");
                #endif

                errorCuda = cudaGraphicsUnregisterResource (cudaInputTexture);
                CUDA_GET_ERROR (errorCuda, "cudaGraphicsUnregisterResource-ftos-input");

            #else 

                #ifdef ATS_ENABLE_RENDER_TO_FILE

                    //  ABOUT
                    // If it's set to nearest or no framebuffer is present do nothing
                    //  so it just copies the pixels.

                    //  IMPORTANT. Already binded.
                    // glBindFramebuffer (GL_FRAMEBUFFER, n);

                    // --- Get data from GPU
                    glReadPixels (0, 0, oImage.width, oImage.height, outputChannelsType, GL_UNSIGNED_BYTE, pixels);
                    GL::GetError ("after-framebuffer-glReadPixels");
                    LOGINFO ("Read image from GPU using OpenGL context.\n")
                    // --- 

                #endif

            #endif

            #ifdef ATS_ENABLE_RENDER_TO_FILE

                #if defined(ATS_ENABLE_FRAMEBUFFER_RENDER) && ATS_OUTPUT_FILTERING_METHOD == ATS_EDGE

                    cudaMemcpy (pixels, output, pixelsSize, cudaMemcpyDeviceToHost);
                    cudaFree (output);

                #endif

                // --- Save image to a file
                    IMAGE::SaveAsPNG (oImage, pixels, ASSET::FILE_OUT);
                    LOGINFO ("Successfully saved the render image.\n")
                // ---

            #endif

            FREE (pixels); MEMORY::EXIT::POP ();
        // ---


        { // --- Cleanup - Window, Render
		    glDeleteVertexArrays (vaosCount, vaos);
		    FREE (vaos); MEMORY::EXIT::POP ();
    	    glDeleteBuffers (xbosCount, xbos); 
		    FREE (xbos); MEMORY::EXIT::POP ();

            #ifndef ATS_ENABLE_RENDER_TO_FILE
                glDeleteTextures (1, &cudaglInteropTexture);
            #endif

            #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER

                glDeleteTextures (ftosCount, ftos);
		        FREE (ftos); MEMORY::EXIT::POP ();
		        glDeleteRenderbuffers (rbosCount, rbos);
                FREE (rbos); MEMORY::EXIT::POP ();
    	        glDeleteFramebuffers (fbosCount, fbos);
                FREE (fbos); MEMORY::EXIT::POP ();

            #endif

		    FREE (uniformsLists); MEMORY::EXIT::POP ();

		    SHADER::DeleteShaders(shadersCount, shaders);
		    FREE (shaders); MEMORY::EXIT::POP ();

		    glDeleteTextures (texturesCount, textures);
		    FREE (textures); MEMORY::EXIT::POP ();

		    glfwTerminate (); MEMORY::EXIT::POP ();
        }

	}


    { // --- Cleanup
        BSTOP ("Finalized Execution!\n")
    } // ---

    
    #ifdef DEBUG_FLAG_CLOCKS
        fprintf (stdout, "Execution Time: [" LOGGER_TIME_FORMAT "]\n", TIMESTAMP::GetElapsed (TIMESTAMP_BEGIN));
    #endif


	return 0;

}
