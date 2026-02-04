# --- Define a 'library' with it's 'headers'
set (GLIB_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/inc)
add_library (GLIB INTERFACE)

# --- Add RELEASE + DEBUG SubProjects
target_link_libraries (GLIB INTERFACE GLAD)
target_link_libraries (GLIB INTERFACE GLM)
target_link_libraries (GLIB INTERFACE WGLEXT)
target_link_libraries (GLIB INTERFACE GLFW)

target_include_directories (GLIB INTERFACE ${GLIB_INC_DIR})
