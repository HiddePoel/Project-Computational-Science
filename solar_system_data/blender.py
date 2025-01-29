import bpy
import sys
import os


def run():
    blend_dir = os.path.dirname(bpy.data.filepath)
    if blend_dir not in sys.path:
        sys.path.append(blend_dir)

    # Import the Bodies module
    from Bodies import Body

    # Constants
    position_scale = 4.5e6  # Scale for positions (4.5 billion km)
    radius_scale = 1e5  # Scale for radii

    sun = Body("Sun", 696340, 1.989e30, get_data=False, color=(255, 255, 0))
    mercury = Body("Mercury", 2439.4, 3.302e23)
    venus = Body("Venus", 6051.84, 48.685e23)
    earth = Body("Earth", 6371.01, 5.97219e24)
    mars = Body("Mars", 3389.92, 6.4171e23)
    jupiter = Body("Jupiter", 69911, 1898.18722e24)
    saturn = Body("Saturn", 58232, 5.6834e26)
    uranus = Body("Uranus", 25559, 86.813e24)
    neptune = Body("Neptune", 24624, 102.409e24)

    bodies = [sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]

    max_frames = 120

    # Clear existing objects except for the camera
    for obj in bpy.data.objects:
        if obj.type != "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Set viewport and render background to black
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (
        0,
        0,
        0,
        1,
    )  # Black

    # Create and animate celestial bodies
    for body in bodies:
        # Create a sphere for the celestial body
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=body.radius / radius_scale, location=(0, 0, 0), segments=64, ring_count=32
        )
        sphere = bpy.context.object

        if body.name.lower() == "earth":
            add_texture(sphere, f"{blend_dir}/textures/earth.jpg")
        # add_texture(sphere, f"{blend_dir}/textures/{body.name.lower()}.jpg")

        body.blender_object = sphere
        sphere.name = body.name

        # Set the sphere's color
        mat = bpy.data.materials.new(name=f"{body.name}_Material")
        mat.diffuse_color = (*[c / 255 for c in body.color], 1)  # Convert RGB to 0-1 scale
        sphere.data.materials.append(mat)

        # Animate if positional data exists
        if body.data.size > 0:
            for frame_idx, position in enumerate(body.data):
                x, y, z = position[:3] / position_scale  # Extract x, y, z coordinates
                sphere.location = (x, y, z)  # Scale positions down to fit
                sphere.keyframe_insert(data_path="location", frame=frame_idx + 1)

                if frame_idx == max_frames:
                    break

    # Add light to the Sun
    bpy.ops.object.light_add(type="POINT", location=(0, 0, 0))
    sun_light = bpy.context.object
    sun_light.name = "Sun_Light"
    sun_light.data.energy = 1000  # Adjust intensity
    sun_light.data.color = (1.0, 1.0, 0.8)  # Slight yellowish color

    # Set render properties
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"  # Use Cycles rendering
    scene.frame_start = 1
    scene.frame_end = max_frames

    # Set output settings
    scene.render.filepath = "//solar_system_animation.mp4"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.audio_codec = "AAC"

    to_render = False

    if to_render:
        # Trigger rendering
        bpy.ops.render.render(animation=True)


def add_texture(sphere, texture_path):
    """Adds an image texture to a sphere in Blender."""
    mat = bpy.data.materials.new(name=f"{sphere.name}_Material")
    mat.use_nodes = True  # Enable Shader Nodes

    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    tex_image = mat.node_tree.nodes.new("ShaderNodeTexImage")

    # Load the texture image
    tex_image.image = bpy.data.images.load(texture_path)

    # Link the texture to the BSDF shader
    mat.node_tree.links.new(bsdf.inputs["Base Color"], tex_image.outputs["Color"])

    # Assign material to sphere
    sphere.data.materials.append(mat)


if __name__ == "__main__":
    run()
