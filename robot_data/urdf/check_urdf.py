import os
import xml.etree.ElementTree as ET

# ==========================================
# 把这里改成你的 URDF 文件名
URDF_FILE = "2022.SLDASM.urdf" 
# ==========================================

def check_urdf_paths():
    if not os.path.exists(URDF_FILE):
        print(f"❌ 错误: 找不到 URDF 文件: {URDF_FILE}")
        return

    print(f"正在检查文件: {URDF_FILE} ...\n")
    
    try:
        tree = ET.parse(URDF_FILE)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ XML 解析失败: {e}")
        print("请检查 URDF 文件是否完整，有没有少尖括号。")
        return

    error_count = 0
    
    # 查找所有的 mesh 标签
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            # 1. 检查 package:// 前缀
            if "package://" in filename:
                print(f"⚠️  警告: 发现 'package://' 前缀: {filename}")
                print("   PyBullet 不支持 package://，请改为相对路径。")
                error_count += 1
                continue

            # 2. 检查反斜杠
            if "\\" in filename:
                print(f"⚠️  警告: 发现 Windows 反斜杠 '\\': {filename}")
                print("   请全部替换为正斜杠 '/'")
                error_count += 1
                
            # 3. 检查文件实际是否存在
            # 路径是相对于 URDF 文件的
            abs_path = os.path.abspath(filename)
            
            if os.path.exists(abs_path):
                print(f"✅ 成功找到: {filename}")
            else:
                print(f"❌ 文件缺失: {filename}")
                print(f"   系统试图寻找: {abs_path}")
                error_count += 1

    print("\n" + "="*30)
    if error_count == 0:
        print("🎉 完美！所有网格路径都正确。如果 PyBullet 还在报错，可能是 STL 文件本身损坏。")
    else:
        print(f"发现 {error_count} 个路径错误。请根据上面的提示修改 URDF 文件。")

if __name__ == "__main__":
    # 确保脚本在当前目录下运行
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    check_urdf_paths()