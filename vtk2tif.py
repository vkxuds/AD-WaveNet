import os
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import tifffile

def convert_vtk_to_tif(vtk_file, output_dir):
    # 读取 vtk 文件
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(vtk_file)
    reader.Update()

    # 获取数据
    data = reader.GetOutput()
    dimensions = data.GetDimensions()
    print(f"Converting {vtk_file}: Dimensions:", dimensions)  # 输出数据的维度

    # 获取点数据中的标量数据
    point_data = data.GetPointData()
    array_0 = point_data.GetArray(0)

    # 将 vtk 数据转换为 NumPy 数组
    numpy_data = vtk_to_numpy(array_0)
    numpy_data = numpy_data.reshape(dimensions, order='F')

    # 检查数据类型
    print("NumPy array data type:", numpy_data.dtype)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 切片、处理并保存为 tif 文件
    for i in range(dimensions[2]):
        slice_data = numpy_data[:, :, i]

        # 将切片数据乘以 10 (如果需要)
        # slice_data = slice_data * 10

        # 保存为 tif 文件
        output_file = os.path.join(output_dir, f'slice_{i:03d}.tif')
        tifffile.imwrite(output_file, slice_data.astype(np.float32))  # 使用 float32 类型保存数据
        print(f'Saved {output_file}')

def batch_convert_vtk_to_tif(input_dir, output_base_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".vtk"):
                vtk_file = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                vtk_filename_without_extension = os.path.splitext(file)[0]
                output_dir = os.path.join(output_base_dir, relative_path, vtk_filename_without_extension)
                convert_vtk_to_tif(vtk_file, output_dir)




if __name__ == "__main__":
    input_dir = "D:/20240828"  # 替换为实际的 VTK 文件目录路径
    output_base_dir = "D:/20240828tif"  # 替换为实际的输出目录路径
    batch_convert_vtk_to_tif(input_dir, output_base_dir)
