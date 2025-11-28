import pybullet as p
import pybullet_data
import time
import os

# ================= 配置 =================
# 这里填你修改后的 URDF 文件路径
# 例如： "my_robot_description/urdf/my_robot.urdf"
ROBOT_URDF_PATH = "robot_data/urdf/2022.SLDASM.urdf" 
# ========================================

def main():
    # 1. 启动仿真
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    # 2. 检查文件是否存在
    if not os.path.exists(ROBOT_URDF_PATH):
        print(f"ERROR: 找不到文件 {ROBOT_URDF_PATH}")
        print("请检查路径是否正确，注意斜杠方向 '/'")
        return

    print("正在加载模型...")
    
    try:
        # 3. 加载 URDF 的关键参数
        # useFixedBase=1: 必须加！否则机械臂会像一堆散架的零件掉在地上
        # flags: 用于处理 SolidWorks 导出的一些不标准网格数据
        robot_id = p.loadURDF(
            ROBOT_URDF_PATH,
            basePosition=[0, 0, 0],
            useFixedBase=1, 
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        )
    except Exception as e:
        print("\n!!!!!!!! 加载失败 !!!!!!!!")
        print("常见原因：")
        print("1. URDF 里的 mesh 路径没改对（一定要去掉 package://）")
        print("2. STL 文件不在对应的路径下")
        print(f"具体错误信息: {e}")
        while True: time.sleep(1) # 保持窗口不关以便看报错
        return

    print("加载成功！")
    
    # 4. 自动创建滑块
    num_joints = p.getNumJoints(robot_id)
    sliders = []
    
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_type = info[2]
        if joint_type != p.JOINT_FIXED:
            name = info[1].decode("utf-8")
            # 读取限位，如果没有限位就给个默认值
            lower = info[8] if info[8] < info[9] else -3.14
            upper = info[9] if info[8] < info[9] else 3.14
            slider = p.addUserDebugParameter(name, lower, upper, 0)
            sliders.append((i, slider))

    # 5. 循环
    while True:
        for joint_idx, slider_id in sliders:
            target_pos = p.readUserDebugParameter(slider_id)
            p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, target_pos)
        
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    main()