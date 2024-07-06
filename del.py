import os

file_path = 'U1652_pytorch_result.mat'
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"已删除文件：{file_path}")
else:
    print(f"文件不存在，不需要删除：{file_path}")
