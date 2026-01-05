from flask import Flask, render_template, request, jsonify
import requests
import os
from PIL import Image
import io
import base64
import zipfile

app = Flask(__name__)

# You need to get the API Key from: https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey
# The AK/SK provided are for account access, not direct API calls
# After logging in with your AK/SK, generate an API Key in the Ark console
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
API_KEY = "f29983c8-351d-427f-8f2b-f89311b372da"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_images', methods=['POST'])
def generate_images():
    data = request.json
    user_prompt = data['prompt']
    prompt = f"生成一个白色背景黑色填充的简笔画风格图像，做成{user_prompt}的形状"
    
    # Call Doubao API to generate image
    response = requests.post(DOUBAO_API_URL, json={
        'model': 'doubao-seedream-4-0-250828',
        'prompt': prompt,
        'size': '1024x1024',
        'response_format': 'b64_json',
        'watermark': False
    }, headers={
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    })
    
    print(f"Request sent - Status: {response.status_code}")
    print(f"Full Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        if 'data' in result and len(result['data']) > 0:
            image_data = result['data'][0]['b64_json']
            return jsonify({'image': image_data})
        else:
            return jsonify({'error': f'No image data in response: {result}'})
    else:
        error_response = response.text
        print(f"ERROR - Status: {response.status_code}")
        print(f"ERROR - Details: {error_response}")
        return jsonify({'error': f'API Error {response.status_code}: {error_response}'})

@app.route('/generate_stl', methods=['POST'])
def generate_stl():
    data = request.json
    layers = data['layers']  # list of base64 images, one per layer
    num_layers = int(data['num_layers'])
    heights = data.get('heights') or []
    positions = data.get('positions') or []
    # Normalize heights to floats with a sane default
    height_values = []
    for i in range(num_layers):
        try:
            val = float(heights[i]) if i < len(heights) else 2.0
            if val <= 0:
                val = 2.0
        except Exception:
            val = 2.0
        height_values.append(val)
    
    stl_files = []
    z_offsets = []
    for idx in range(num_layers):
        if idx == 0:
            z_offsets.append(0.0)
            continue
        placement = positions[idx] if idx < len(positions) else "stack"
        if placement == "same":
            z_offsets.append(z_offsets[idx - 1])
        else:
            z_offsets.append(z_offsets[idx - 1] + height_values[idx - 1])
    
    try:
        for layer_idx, layer_image_b64 in enumerate(layers):
            # Decode base64 image
            image_data = base64.b64decode(layer_image_b64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to grayscale and get black pixels
            image = image.convert('L')
            width, height = image.size
            pixels = image.load()
            
            # Find black pixels (threshold < 50)
            black_points = []
            for y in range(height):
                for x in range(width):
                    if pixels[x, y] < 50:  # Black pixel
                        black_points.append((x, y))
            
            if len(black_points) == 0:
                return jsonify({'error': f'No black pixels found in layer {layer_idx + 1}'})
            
            # Generate STL for this layer; z_offset from previous layer heights
            z_offset = z_offsets[layer_idx]
            thickness = height_values[layer_idx]
            stl_content = generate_stl_from_points(black_points, width, height, z_offset, thickness)
            
            stl_files.append({
                'name': f'layer_{layer_idx + 1}.stl',
                'content': stl_content
            })
        
        # Create ZIP file containing all STL files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for stl_file in stl_files:
                zip_file.writestr(stl_file['name'], stl_file['content'])
        
        zip_buffer.seek(0)
        zip_b64 = base64.b64encode(zip_buffer.read()).decode('utf-8')
        
        return jsonify({'zip_file': zip_b64, 'num_layers': num_layers})
    
    except Exception as e:
        return jsonify({'error': str(e)})

def generate_stl_from_points(points, width, height, z_offset, thickness):
    """Generate STL file from 2D black points by extruding them into 3D.
    x/y positions stay consistent with source image; layers are stacked by provided heights.
    """
    scale = 0.1  # mm per pixel

    solid_name = "layer"
    stl = f"solid {solid_name}\n"
    step = 2  # voxel sampling to reduce file size
    
    voxels = set((x // step, y // step) for x, y in points)
    
    for vox_x, vox_y in voxels:
        x = vox_x * step
        y = vox_y * step
        
        x3d = x * scale
        y3d = (height - y) * scale  # Flip Y axis to prevent mirroring
        z_bottom = z_offset
        z_top = z_offset + thickness
        size = step * scale
        
        # Create a cube for this voxel (12 triangles, 6 faces)
        # Bottom face (z = z_bottom)
        stl += create_triangle(
            [x3d, y3d, z_bottom],
            [x3d + size, y3d, z_bottom],
            [x3d + size, y3d + size, z_bottom]
        )
        stl += create_triangle(
            [x3d, y3d, z_bottom],
            [x3d + size, y3d + size, z_bottom],
            [x3d, y3d + size, z_bottom]
        )
        
        # Top face (z = z_top)
        stl += create_triangle(
            [x3d, y3d, z_top],
            [x3d + size, y3d + size, z_top],
            [x3d + size, y3d, z_top]
        )
        stl += create_triangle(
            [x3d, y3d, z_top],
            [x3d, y3d + size, z_top],
            [x3d + size, y3d + size, z_top]
        )
        
        # Front face
        stl += create_triangle(
            [x3d, y3d, z_bottom],
            [x3d + size, y3d, z_bottom],
            [x3d + size, y3d, z_top]
        )
        stl += create_triangle(
            [x3d, y3d, z_bottom],
            [x3d + size, y3d, z_top],
            [x3d, y3d, z_top]
        )
        
        # Back face
        stl += create_triangle(
            [x3d, y3d + size, z_bottom],
            [x3d + size, y3d + size, z_top],
            [x3d + size, y3d + size, z_bottom]
        )
        stl += create_triangle(
            [x3d, y3d + size, z_bottom],
            [x3d, y3d + size, z_top],
            [x3d + size, y3d + size, z_top]
        )
        
        # Left face
        stl += create_triangle(
            [x3d, y3d, z_bottom],
            [x3d, y3d, z_top],
            [x3d, y3d + size, z_top]
        )
        stl += create_triangle(
            [x3d, y3d, z_bottom],
            [x3d, y3d + size, z_top],
            [x3d, y3d + size, z_bottom]
        )
        
        # Right face
        stl += create_triangle(
            [x3d + size, y3d, z_bottom],
            [x3d + size, y3d + size, z_top],
            [x3d + size, y3d, z_top]
        )
        stl += create_triangle(
            [x3d + size, y3d, z_bottom],
            [x3d + size, y3d + size, z_bottom],
            [x3d + size, y3d + size, z_top]
        )
    
    stl += f"endsolid {solid_name}\n"
    return stl

def create_triangle(v1, v2, v3):
    """Create a triangle facet in STL ASCII format with computed normal"""
    # Compute normal via cross product of (v2 - v1) x (v3 - v1)
    ax, ay, az = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
    bx, by, bz = v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    length = (nx * nx + ny * ny + nz * nz) ** 0.5
    if length == 0:
        normal = (0.0, 0.0, 0.0)
    else:
        normal = (nx / length, ny / length, nz / length)
    
    return (
        f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n"
        f"    outer loop\n"
        f"      vertex {v1[0]} {v1[1]} {v1[2]}\n"
        f"      vertex {v2[0]} {v2[1]} {v2[2]}\n"
        f"      vertex {v3[0]} {v3[1]} {v3[2]}\n"
        f"    endloop\n"
        f"  endfacet\n"
    )

if __name__ == '__main__':
    app.run(debug=True, port=8080)