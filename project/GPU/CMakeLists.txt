# --- Define a 'subproject'
project (GPU VERSION 1.0.0 LANGUAGES C CXX CUDA)

# --- Add header files
file (GLOB_RECURSE HEADER_FILES
	${PROJECT_SOURCE_DIR}/inc/*.h
	${PROJECT_SOURCE_DIR}/inc/*.hpp
)

# --- Add needed DEFINE's
add_compile_definitions (_CRT_SECURE_NO_WARNINGS)

# --- Info Build Type
message (STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if 		(${PROFILE} EQUAL 1)

	message (STATUS "Preset: CPU - DEBUG")
	add_compile_definitions (DEBUG_TYPE=${DEBUG_FLAG_LOGGING}+${DEBUG_FLAG_MEMORY}+${DEBUG_FLAG_CLOCKS})

elseif 	(${PROFILE} EQUAL 2)

	message (STATUS "Preset: CPU - RELEASE")
	add_compile_definitions (DEBUG_TYPE=${DEBUG_FLAG_CLOCKS})

endif ()


# --- Define the executable
add_executable (${PROJECT_NAME} ${HEADER_FILES} 
	src/main.cu 
	src/stb_image_implementation.cpp
)

# --- LIBS For shared by all presets.
target_link_libraries (${PROJECT_NAME} BLUELIB)
target_link_libraries (${PROJECT_NAME} CPULIB)
target_link_libraries (${PROJECT_NAME} GLIB)
target_link_libraries (${PROJECT_NAME} STB)
target_link_libraries (${PROJECT_NAME} cuda cudart)

# --- By default the CUDA compiler uses whole-program compilation. 
# ---  Effectively this means that all device functions and variables needed to be located inside a 
# ---  single file or compilation unit.
set_property (TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# ---
set_property (TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# ---
target_include_directories (${PROJECT_NAME} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# --- Creates a 'symlink' for resources in builds directory.
add_custom_command (
	TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink
	${CMAKE_SOURCE_DIR}/project/${PROJECT_NAME}/res
	${CMAKE_CURRENT_BINARY_DIR}/res
)

# --- Creates a 'symlink' for output folder in builds directory.
add_custom_command (
	TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink
	${CMAKE_SOURCE_DIR}/out
	${CMAKE_CURRENT_BINARY_DIR}/out
)

# --- Copy necessary .dll's
add_custom_command ( TARGET ${PROJECT_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		$<TARGET_FILE:GLFW> 
		$<TARGET_FILE_DIR:${PROJECT_NAME}>
)
