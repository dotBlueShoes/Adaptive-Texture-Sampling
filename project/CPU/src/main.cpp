// Made by Matthew Strumillo 2024.07.20

#include "heading.hpp"
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


s32 main (s32 argumentsCount, c8* arguments[]) {

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


    // --- Args interpreter  
        c8* imageFilepath;
        c8* specialFilepath;
        u8 imageChannels;

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
                imageChannels = 3;
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
                imageChannels = atoi(arguments[iChannels]);
            } break;

            default:
                LOGWARN ("Invalid number of arguments passed!\n");
            case 1: { // Apply DEFAULT values for arguments.
                imageFilepath   = (c8*)(void*)ASSET::FILE_I0_NOALPHA;
                specialFilepath = (c8*)(void*)ASSET::FILE_I0_UPSCALE;
                imageChannels = 3;
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
            // MSAA_LEVEL passed here is not checked with the possible maximum!
            //  The check however does happen for FRAMEBUFFER use. 
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
        

        { // Properly deallocate data if we hit ERROR.
            // TODO MEMORY::EXIT::PUSH (IMAGE::Free, iImage.data);
            MEMORY::EXIT::PUSH (GLFW_TERMINATE, nullptr);
        }

        if (window == nullptr) ERROR ("Failed to create GLFW window!\n");

        // Bind GL Context to Window.
        glfwMakeContextCurrent (window);

        // More GLFW initialization.
        glfwSetFramebufferSizeCallback (window, framebuffer_size_callback);

        #ifdef ATS_SWAP_INTERVAL
            glfwSwapInterval (ATS_SWAP_INTERVAL);
        #endif

        // Initialize GLAD.
        if (!gladLoadGLLoader ((GLADloadproc)glfwGetProcAddress)) {
            ERROR ("Failed to initialize GLAD!\n");
        }

        #ifdef ATS_ENABLE_EXTENSIONS_CHECK
            LOGINFO ("GL device: %s\n", glGetString(GL_VENDOR));
            GL::FunctionalityCheck ();
        #endif

    } // ---


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

            #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER
                // Material->Shader 's
                u32 shadersCount = 2;           
                GLuint* shaders;
            #else
                // Material->Shader 's
                u32 shadersCount = 1;           
                GLuint* shaders;
            #endif
            
            #ifdef ATS_ENABLE_CMAA2
                // Compute shaders
                u32 computesCount = 1;           
                GLuint* computes;
            #endif

            // Material->Shader->Uniforms 's
            #ifdef ATS_ENABLE_FRAMEBUFFER_DEPTH_TEXTURE
                u32 uniformsListsCount = 11; // + 2
                u16* uniformsLists;             
            #else
                #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER
                    u32 uniformsListsCount = 10; // + 2
                    u16* uniformsLists;  
                #else
                    u32 uniformsListsCount = 6;     
                    u16* uniformsLists;  
                #endif
            #endif

            // Material->Texture 's
            u32 texturesCount = 2;          
            GLuint* textures;	

            #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER

                // Framebuffer 's
                u32 fbosCount = 1;               
                GLuint* fbos;

                #ifdef ATS_ENABLE_FRAMEBUFFER_DEPTH_TEXTURE

                    // Framebuffer->Texture 's
                    u32 ftosCount = 3; // + 1          
                    GLuint* ftos;	


                #else 

                    // Framebuffer->Texture 's
                    u32 ftosCount = 2;          
                    GLuint* ftos;	

                    // Framebuffer->RenderBuffer 's
                    u32 rbosCount = 1;               
                    GLuint* rbos;

                #endif

            #endif

        // ---


        // --- Rendererer scene and environment setting up.

            { // --- Settings

                // { // --- Activate Texture Handlers
                //     //
                //     //glActiveTexture (GL_TEXTURE1);
                // }

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


            { // Activate required (used) Texture Handlers
                // [screen] framebuffer output texture or [world] diffuse texture.
                glActiveTexture (GL_TEXTURE0);

                // [screen] depth texture or [world] Upscale texture.
                glActiveTexture (GL_TEXTURE1);
            }


            { // --- Load Textures.

                ALLOCATE (GLuint, textures, texturesCount * sizeof (GLuint));
                MEMORY::EXIT::PUSH (FREE, textures);

                glGenTextures (texturesCount, textures);

                #if ATS_TEXTURE_FILTERING_METHOD == ATS_NEAREST

                    #if ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_NONE

                        TEXTURE::Create (textures[0], iImage, iImageData, GL_NEAREST, GL_NEAREST);
                        TEXTURE::Create (textures[1], specialImage, specialImageData, GL_NEAREST, GL_NEAREST);

                    #elif ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_NEAREST

                        TEXTURE::CreateMipmapped (textures[0], iImage, iImageData, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST);

                    #elif ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_LINEAR

                        TEXTURE::CreateMipmapped (textures[0], iImage, iImageData, GL_LINEAR_MIPMAP_NEAREST, GL_NEAREST);
                        TEXTURE::CreateMipmapped (textures[1], specialImage, specialImageData, GL_LINEAR_MIPMAP_NEAREST, GL_NEAREST);

                    #endif

                #elif ATS_TEXTURE_FILTERING_METHOD == ATS_LINEAR

                    #if ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_NONE

                        TEXTURE::Create (textures[0], iImage, iImageData, GL_NEAREST, GL_LINEAR);
                        TEXTURE::Create (textures[1], specialImage, specialImageData, GL_NEAREST, GL_LINEAR);

                    #elif ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_NEAREST

                        TEXTURE::CreateMipmapped (textures[0], iImage, iImageData, GL_NEAREST_MIPMAP_NEAREST, GL_LINEAR);

                    #elif ATS_TEXTURE_MIPMAP_FILTERING_METHOD == ATS_MIPMAP_LINEAR

                        TEXTURE::CreateMipmapped (textures[0], iImage, iImageData, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR);
                        TEXTURE::CreateMipmapped (textures[1], specialImage, specialImageData, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR);

                    #endif

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


            #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER

                { // --- Create Framebuffer Textures.
                    ALLOCATE (GLuint, ftos, ftosCount * sizeof (GLuint));
                    MEMORY::EXIT::PUSH (FREE, ftos);

                    glGenTextures (ftosCount, ftos);

                    #ifdef ATS_ENABLE_CMAA2
                        const auto inputformat = GL_RGBA8;
                        const auto outputformat = GL_RGBA;
                    #else
                        const auto inputformat = GL_RGB8;
                        const auto outputformat = GL_RGB;
                    #endif

                    #if ATS_FRAMEBUFFER_FILTERING_METHOD == ATS_NEAREST
                        const auto filtering = GL_NEAREST;
                    #elif ATS_FRAMEBUFFER_FILTERING_METHOD == ATS_LINEAR
                        const auto filtering = GL_LINEAR;
                    #endif


                    #if (ATS_MSAA_LEVEL == 0)

                        FRAMEBUFFER::TEXTURE::Create (
                            ftos[0], inputformat, outputformat,
                            ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y, 
                            filtering, GL_UNSIGNED_BYTE
                        );

                        FRAMEBUFFER::TEXTURE::Create (
                            ftos[1], inputformat, outputformat,
                            ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y, 
                            filtering, GL_UNSIGNED_BYTE
                        );

                    #else

                        FRAMEBUFFER::MSAA::TEXTURE::Create (
                            ftos[0], inputformat, outputformat,
                            ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y
                        );
                        
                    #endif

                } // ---


                #ifdef ATS_ENABLE_FRAMEBUFFER_DEPTH_TEXTURE

                    // { // All depth textures from the patch are associated with the 2nd SAMPLER ! 
                    //     glActiveTexture (GL_TEXTURE1);
                    // }

                    FRAMEBUFFER::TEXTURE::Create (
                        ftos[2], GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT,
                        ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y, 
                        GL_NEAREST, GL_UNSIGNED_INT
                    );

                    // GL_FLOAT
                    // GL_UNSIGNED_INT

                #else

                    { // --- Create Render Buffer Object -> Stencil & Depth
                        ALLOCATE (GLuint, rbos, rbosCount * sizeof (GLuint));
                        MEMORY::EXIT::PUSH (FREE, rbos);

                        glGenRenderbuffers (rbosCount, rbos);

                        #if (ATS_MSAA_LEVEL == 0)
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

                #endif


                { // --- Create Framebuffer
                    ALLOCATE (GLuint, fbos, fbosCount * sizeof (GLuint));
                    MEMORY::EXIT::PUSH (FREE, fbos);

                    glGenFramebuffers (fbosCount, fbos);

                    #ifdef ATS_ENABLE_FRAMEBUFFER_DEPTH_TEXTURE

                        #if (ATS_MSAA_LEVEL == 0)
                            FRAMEBUFFER::DEPTHFRAMEBUFFER::Create (
                                fbos[0], 
                                ftos[0], GL_COLOR_ATTACHMENT0, 
                                ftos[1], GL_COLOR_ATTACHMENT1, 
                                ftos[2], GL_DEPTH_ATTACHMENT
                            );
                        #else
                            FRAMEBUFFER::DEPTHFRAMEBUFFER::MSAA::Create (
                                fbos[0], 
                                ftos[0], GL_COLOR_ATTACHMENT0, 
                                ftos[1], GL_DEPTH_ATTACHMENT
                            ); 
                        #endif

                    #else
                        #if (ATS_MSAA_LEVEL == 0)
                            FRAMEBUFFER::Create (
                                fbos[0], 
                                ftos[0], GL_COLOR_ATTACHMENT0,
                                ftos[1], GL_COLOR_ATTACHMENT1, 
                                rbos[0], GL_DEPTH_STENCIL_ATTACHMENT
                            ); 
                        #else
                            FRAMEBUFFER::MSAA::Create (
                                fbos[0], 
                                ftos[0], GL_COLOR_ATTACHMENT0, 
                                rbos[0], GL_DEPTH_STENCIL_ATTACHMENT
                            ); 
                        #endif
                    #endif
                    
                } // ---

            #endif


            //LOGINFO ("pre-shader\n");


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

                    #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER

                        { // - shader 1
                            SHADER::ReadFile (ASSET::SHADER_VERT_FRAMEBUFFER, bufferVert, ATS_SHADER_VERT);
                            SHADER::ReadFile (ASSET::SHADER_FRAG_FRAMEBUFFER, bufferFrag, ATS_SHADER_FRAG);
                        
                            SHADER::Compile (shaders[1], bufferVert, bufferFrag);
                        }

                    #endif
                    
                }

                #ifdef ATS_GLSL_INJECTION

                    FREE (GLSL::ffsd); MEMORY::EXIT::POP ();

                #endif

                FREE (buffer); MEMORY::EXIT::POP ();
            } // ---


            #ifdef ATS_ENABLE_CMAA2
            {
                ALLOCATE (GLuint, computes, computesCount * sizeof (GLuint));
                MEMORY::EXIT::PUSH (FREE, computes);

                const u16 bufferSize = ATS_MAX_CHARS_FOR_SHADER_COMP_FILE;
                c8* buffer;

                ALLOCATE (c8, buffer, bufferSize * sizeof (c8));
                MEMORY::EXIT::PUSH (FREE, buffer);

                SHADER::ReadFile (ASSET::SHADER_COMP, buffer, ATS_SHADER_COMP);
                SHADER::COMPUTE::Compile (computes[0], buffer);
                
                FREE (buffer); MEMORY::EXIT::POP ();
            }
            #endif


            //LOGINFO ("post-shader\n");


            // --- Set Shaders uniforms
                ALLOCATE (u16, uniformsLists, uniformsListsCount * sizeof (u16));
                MEMORY::EXIT::PUSH (FREE, uniformsLists);

                #ifdef ATS_ENABLE_FRAMEBUFFER_DEPTH_TEXTURE

                    //  IMPORTANT
                    // Remember to manually change the SHADER_N_UNIFORMS variables when changing shaders code!
                    //

                    const u32 SHADER_FINAL_UNIFORMS = 0;
                    const u32 SHADER_SCENE_UNIFORMS = 4;

                    uniformsLists[0] = 3;
            
                    #if defined(ATS_MSAA_DISABLE_MANUAL_RESOLVE) || (ATS_MSAA_LEVEL == 0)
                        uniformsLists[1] = UNIFORM::ID::TEXTURE0;
                    #else
                        uniformsLists[1] = UNIFORM::ID::MULTISAMPLE_TEXTURE;
                    #endif

                    uniformsLists[2] = UNIFORM::ID::TEXTURE1;
                    uniformsLists[3] = UNIFORM::ID::TEXTURE2;

                    uniformsLists[4] = 6;
                    uniformsLists[5] = UNIFORM::ID::COLOR;      // layout 0 
                    uniformsLists[6] = UNIFORM::ID::TEXTURE0;   // layout 1 
                    uniformsLists[7] = UNIFORM::ID::PROJECTION; // layout 2 
                    uniformsLists[8] = UNIFORM::ID::VIEW;       // layout 3 
                    uniformsLists[9] = UNIFORM::ID::TRANSFORM;  // layout 4 
                    uniformsLists[10] = UNIFORM::ID::TEXTURE1;  // layout 5

                #else

                    //  IMPORTANT
                    // Remember to manually change the SHADER_N_UNIFORMS variables when changing shaders code!
                    //

                    #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER
                        const u32 SHADER_FINAL_UNIFORMS = 0;
                        const u32 SHADER_SCENE_UNIFORMS = 3;

                        uniformsLists[0] = 2;
            
                        #if defined(ATS_MSAA_DISABLE_MANUAL_RESOLVE) || (ATS_MSAA_LEVEL == 0)
                            uniformsLists[1] = UNIFORM::ID::TEXTURE0; // layout 0 
                        #else
                            uniformsLists[1] = UNIFORM::ID::MULTISAMPLE_TEXTURE; // layout 0 
                        #endif

                        uniformsLists[2] = UNIFORM::ID::TEXTURE1;   // layout 1 

                        uniformsLists[3] = 6;
                        uniformsLists[4] = UNIFORM::ID::COLOR;      // layout 0 
                        uniformsLists[5] = UNIFORM::ID::TEXTURE0;   // layout 1
                        uniformsLists[6] = UNIFORM::ID::PROJECTION; // layout 2
                        uniformsLists[7] = UNIFORM::ID::VIEW;       // layout 3
                        uniformsLists[8] = UNIFORM::ID::TRANSFORM;  // layout 4
                        uniformsLists[9] = UNIFORM::ID::TEXTURE1;   // layout 5
                    #else
                        const u32 SHADER_SCENE_UNIFORMS = 0;
            
                        uniformsLists[0] = 5;
                        uniformsLists[1] = UNIFORM::ID::COLOR;
                        uniformsLists[2] = UNIFORM::ID::TEXTURE0;
                        uniformsLists[3] = UNIFORM::ID::PROJECTION;
                        uniformsLists[4] = UNIFORM::ID::VIEW;
                        uniformsLists[5] = UNIFORM::ID::TRANSFORM;
                    #endif

                #endif
                
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
                    auto&& transform = *(glm::mat4*)(UNIFORM::transform + 1);
                    auto&& count = *(GLint*)UNIFORM::transform;

                    // NOTE. We're presetting TRANSFORM to identity matrix later.

                    count = 1;
                }

            } // ---

        } // ---


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

                { // TEXTURE Diffuse
                    UNIFORM::texture0[0] = 0;
                    UNIFORM::texture0[1] = textures[0];
                }

                { // TEXTURE Upscale
                    UNIFORM::texture1[0] = 1;
                    UNIFORM::texture1[1] = textures[1];
                }

                { // COLOR
                    //  TODO
                    // - MAGIC_NUMBERS. Changing the GREEN value.
                    // - This is old code. Resolve if it is even used. Don't obfuscate execution time.
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

                //
                // LOGINFO ("- framebuffer-render\n");
                //

                #ifdef ATS_ENABLE_CMAA2

                    // issues 
                    // - ftos[0] - is not rgba8 it's rgb8.
                    
                    glBindImageTexture (
                        0,            // image unit
                        ftos[0],      // texture object
                        0,            // mip level
                        GL_FALSE,     // layered (for 3D or array textures)
                        0,            // layer
                        GL_READ_WRITE,// access: GL_READ_ONLY / GL_WRITE_ONLY / GL_READ_WRITE
                        GL_RGBA8      // format (must match internal format)
                    );

                    glUseProgram (computes[0]);
                    glDispatchCompute (ATS_ASSET_OUTPUT_CANVAS_X / 8, ATS_ASSET_OUTPUT_CANVAS_Y / 4, 1);
                    glMemoryBarrier (GL_ALL_BARRIER_BITS);

                #endif

                { // --- Default framebuffer render.

                    glViewport (0, 0, ATS_ASSET_OUTPUT_CANVAS_X, ATS_ASSET_OUTPUT_CANVAS_Y);

                    //  TODO. MSAA AUTOMATIC RESOLVE
                    // I believe another framebuffer is required so that MSAA_input -> MSAA_output -> post-process_framebuffer 
                    //
                    // glBindFramebuffer (GL_READ_FRAMEBUFFER, fbos[0]);
                    // glBindFramebuffer (GL_DRAW_FRAMEBUFFER, 0);
                    // glBlitFramebuffer (
                    //     0, 0, ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y,
                    //     0, 0, ATS_ASSET_OUTPUT_CANVAS_X, ATS_ASSET_OUTPUT_CANVAS_Y,
                    //     GL_COLOR_BUFFER_BIT,   // color only
                    //     GL_NEAREST             // NEAREST or LINEAR)
                    // );

                    glBindFramebuffer (GL_FRAMEBUFFER, 0);

                    glDisable (GL_DEPTH_TEST);
                    glClear (GL_COLOR_BUFFER_BIT);

                    SHADER::Use (shaders[1]);

                    { // TEXTURE diffuse
                        UNIFORM::texture0[0] = 0;
                        UNIFORM::texture0[1] = ftos[0];
                    }

                    { // TEXTURE upscale
                        UNIFORM::texture1[0] = 1;
                        UNIFORM::texture1[1] = ftos[1];
                    }

                    #ifdef ATS_ENABLE_FRAMEBUFFER_DEPTH_TEXTURE

                        { // DEPTH TEXTURE
                            UNIFORM::texture2[0] = 2;
                            UNIFORM::texture2[1] = ftos[2];
                        }
                        
                    #endif

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


            #ifdef ATS_ENABLE_RENDER_TO_FILE
                break; 
            #endif
            

            // --- Apply and prepere for next.
                glfwSwapBuffers (window);
                glfwPollEvents ();

                glViewport (0, 0, ATS_ASSET_INPUT_CANVAS_X, ATS_ASSET_INPUT_CANVAS_Y);
            // ---

            prevTimeValue = timeValue;
            timeValue = glfwGetTime ();

        } // ---


        #ifdef ATS_ENABLE_RENDER_TO_FILE

            { // --- Saving the framebuffer as image

                const GLsizei height = ATS_ASSET_OUTPUT_CANVAS_Y;
                const GLsizei width = ATS_ASSET_OUTPUT_CANVAS_X;
                const GLenum channelsType = GL_RGBA;
                const u8 channels = 4;

                u8* pixels; ALLOCATE (u8, pixels, width * height * channels);
                MEMORY::EXIT::PUSH (FREE, pixels);

                //  IMPORTANT. Already binded (the right one no matter what).
                // glBindFramebuffer (GL_FRAMEBUFFER, n);


                // --- Get data from GPU
                    glReadPixels (0, 0, width, height, channelsType, GL_UNSIGNED_BYTE, pixels);
                    GL::GetError ("after-framebuffer-glReadPixels");
                    LOGINFO ("Read image from GPU using OpenGL context.\n")
                // --- 


                #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER
                    // --- Flip image vertically (OpenGL origin is bottom-left, stb_image origin is top-left)
                        for (u32 y = 0; y < height / 2; ++y) {
                            for (u32 x = 0; x < width * 4; ++x) {
                                std::swap (pixels[y * width * 4 + x], pixels[(height - 1 - y) * width * 4 + x]);
                            }
                        }
                        LOGINFO ("Swapped texture bottom-top.\n");
                    // ---
                #endif


                // --- Save image to a file
                    IMAGE::Head oImage { channels, height, width };
                    IMAGE::SaveAsPNG (oImage, pixels, ASSET::FILE_OUT);
                    LOGINFO ("Successfully saved the render image.\n")
                // ---


                FREE (pixels); MEMORY::EXIT::POP ();

            } // ---

        #endif


        { // --- Cleanup - Window, Render
            glDeleteVertexArrays (vaosCount, vaos);
            FREE (vaos); MEMORY::EXIT::POP ();
            glDeleteBuffers (xbosCount, xbos); 
            FREE (xbos); MEMORY::EXIT::POP ();

            #ifdef ATS_ENABLE_FRAMEBUFFER_RENDER

                glDeleteTextures (ftosCount, ftos);
                FREE (ftos); MEMORY::EXIT::POP ();

                #ifndef ATS_ENABLE_FRAMEBUFFER_DEPTH_TEXTURE

                    glDeleteRenderbuffers (rbosCount, rbos);
                    FREE (rbos); MEMORY::EXIT::POP ();

                #endif

                glDeleteFramebuffers (fbosCount, fbos);
                FREE (fbos); MEMORY::EXIT::POP ();

            #endif

            FREE (uniformsLists); MEMORY::EXIT::POP ();

            #ifdef ATS_ENABLE_CMAA2
                SHADER::DeleteShaders (computesCount, computes);
                FREE (computes); MEMORY::EXIT::POP ();
            #endif

            SHADER::DeleteShaders (shadersCount, shaders);
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
