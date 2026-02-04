include (CPM)

### LIBARIES

# --- BLUELIB
if (NOT ${BLUELIB_LOCALLY})

	message (STATUS "DISABLED - Locally library BLUE")
	
	FetchContent_Declare (
		BLUELIB
		GIT_REPOSITORY https://github.com/dotBlueShoes/blue.git
		GIT_TAG master
	)
	
	FetchContent_MakeAvailable (BLUELIB)

else ()

	message (STATUS "ENABLED - Locally library BLUE")
	
	# --- Define 'bluelib'
	set (BLUELIB_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/bluelib/include)
	add_library (BLUELIB INTERFACE)
	target_include_directories (BLUELIB INTERFACE ${BLUELIB_INC_DIR})

endif ()

# --- STB
set (STB_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/stb)
add_library (STB INTERFACE)
target_include_directories (STB INTERFACE ${STB_INC_DIR})

# --- GLAD
set (GLAD_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/glad/include)
set (GLAD_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/glad/src)
add_library (GLAD STATIC ${GLAD_SRC_DIR}/glad.c)
target_include_directories (GLAD PUBLIC ${GLAD_INC_DIR})

# --- WGLEXT
set (WGLEXT_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/wglext/include)
add_library (WGLEXT INTERFACE)
target_include_directories (WGLEXT INTERFACE ${WGLEXT_INC_DIR})

# --- GLM
set(GLM_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/glm/1.0.1)
add_library(GLM INTERFACE)
target_include_directories(GLM INTERFACE ${GLM_INC_DIR})

# GLFW
set(GLFW_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/glfw/include)
add_library(GLFW SHARED IMPORTED GLOBAL)
set_property(TARGET GLFW PROPERTY IMPORTED_IMPLIB_RELEASE ${CMAKE_CURRENT_LIST_DIR}/glfw/lib/glfw3.lib)
set_property(TARGET GLFW PROPERTY IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_LIST_DIR}/glfw/bin/glfw3.dll)
set_property(TARGET GLFW PROPERTY IMPORTED_IMPLIB_DEBUG ${CMAKE_CURRENT_LIST_DIR}/glfw/lib/glfw3.lib)
set_property(TARGET GLFW PROPERTY IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_LIST_DIR}/glfw/bin/glfw3.dll)
target_include_directories(GLFW INTERFACE ${GLFW_INC_DIR})
