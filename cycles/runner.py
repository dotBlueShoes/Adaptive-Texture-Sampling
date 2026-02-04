import math
import bpy
import sys
import os

# --- args

if "--" in sys.argv:
    idx = sys.argv.index("--")
    custom_args = sys.argv[idx + 1:]
else:
    custom_args = []

# --- args

bpy.context.scene.world.use_nodes = True

OBJECT_CUBE_NAME = "Cube"
MATERIAL_A_NAME = "image"

blend_dir = bpy.path.abspath("//")

# ---

if len(custom_args) < 1:
    # "C:/Projects/ATS-GPU/cycles/textures/0_HR.png"
    image_path = "textures/0_HR.png"
else:
    image_path = custom_args[0]

image_path = os.path.join(blend_dir, image_path)
image_path = os.path.normpath(image_path)

background_color = (0.033105, 0.072272, 0.072272, 1.000)
object_position = (0.0, 0.0, 0.0)
object_rotation = (math.radians(45.0), math.radians(45.0), math.radians(20.0))

# ---


print ("Running 'runner.py'")

world = bpy.data.worlds.get("World")
obj = bpy.data.objects.get(OBJECT_CUBE_NAME)
mat = bpy.data.materials.get(MATERIAL_A_NAME)


world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")

if bg_node:
    
    bg_node.inputs[0].default_value = background_color
    
else:
    
    print("Could not find the Background node of scene World")
    
    

if obj:

    obj.location = object_position
    obj.rotation_euler = object_rotation
    
else:
    
    print("Could not find node called 'Cube'.\n");
    
    
if mat:
    
    material_nodes = mat.node_tree.nodes

    #
    # Find the image texture node from nodes.
    #
    
    material_node_image = None
    
    for node in material_nodes:
        if node.type == 'TEX_IMAGE':
            material_node_image = node
            break
    
    if material_node_image:
        
        # LOAD
        image = bpy.data.images.load(image_path, check_existing=True)
        
        # SET
        material_node_image.image = image
        
    else:
        
        print("No TEX_IMAGE node inside material.\n");
    
else:
    
    print("Could not find material called 'image'.\n");
    
#  NOTE -> This is not needed  
# bpy.context.view_layer.update()
#

# 
# EXPLICITLY GIVE COMMAND TO SAVE THE CHANGES
#

bpy.ops.wm.save_mainfile()