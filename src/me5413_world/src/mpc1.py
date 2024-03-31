import casadi as ca
import numpy as np
import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, Quaternion
import tf
import numpy as np

# 差速小车模型
dt = 0.1  # 时间步长
x = ca.SX.sym('x', 3)  # 状态量: x, y, yaw
u = ca.SX.sym('u', 2)  # 控制量: u(前进速度), omega(角速度)  
v = u[0]
omega = u[1]

xdot = ca.vertcat(v * ca.cos(x[2]),
                  v * ca.sin(x[2]), 
                  omega)  

f = ca.Function('f', [x, u], [x + xdot * dt])

# MPC参数设置  
N_horizon = 59             # 预测步长
Q_np = np.diag([1, 1, 0.1])   # 状态量权重(Numpy数组)
R_np = np.diag([0.1, 0.1])    # 控制量权重(Numpy数组)

Q = ca.SX(Q_np)  # 将Numpy数组转换为CasADI SX类型
R = ca.SX(R_np)


# 符号变量 
x_k = ca.SX.sym('x0', 3)       # 初始状态
p = ca.SX.sym('p_ref', N_horizon+1, 3) # 参考路径(参数) 
u = ca.SX.sym('u', N_horizon, 2) # 控制序列(优化变量)

# 填充 MPC 优化问题
obj = 0  
constr = []
x_next = x_k

for i in range(N_horizon):
    state_err = x_next.T - p[i,:]

    obj += ca.mtimes([state_err, Q, state_err.T]) + ca.mtimes([u[i,:], R, u[i,:].T])
        
    x_next = f(x_next, u[i,:]) 
    constr += [x_next]
    
# 末端权重
# obj += ca.mtimes([(state_err - p[-1,:]), 10*Q, (state_err - p[-1,:]).T])
    
u_dm = ca.DM(u)
nlp = {'x':u.reshape((-1, 1)), 'f':obj, 'g':ca.vertcat(*constr), 'p': ca.vertcat(x_k.T, p)}
opts = {'ipopt.print_level': 0} 
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# ROS节点初始化
rospy.init_node('mpc_node')

# 订阅路径话题
path_msg = None
def path_callback(msg):
    global path_msg
    path_length = len(msg.poses)
    path_msg = np.zeros((path_length, 3))
    for i, pose in enumerate(msg.poses):
        path_msg[i, 0] = pose.pose.position.x
        path_msg[i, 1] = pose.pose.position.y
        quat = pose.pose.orientation
        path_msg[i, 2] = np.arctan2(2*(quat.w*quat.z + quat.x*quat.y), 
                                    1 - 2*(quat.y**2 + quat.z**2))
        
path_sub = rospy.Subscriber('/me5413_world/planning/local_path', Path, path_callback)

cur_pose = None
def odom_callback(msg):
    global cur_pose
    pos = msg.pose.pose.position
    quat = msg.pose.pose.orientation
    cur_pose = np.array([pos.x, pos.y, tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]])

odom_sub = rospy.Subscriber('/gazebo/ground_truth/state', Odometry, odom_callback)

# 控制命令发布者
cmd_vel_pub = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=1)

# MPC主循环
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    if path_msg is not None and cur_pose is not None:
        x0 = cur_pose  # 初始状态为当前位置

        # res = solver(x0=x0, p=path_msg, lbx=-1, ubx=1)
        # path_ = np.concatenate((x0.T, path_msg), axis=0)
        args = {'x0': np.zeros((N_horizon * 2, 1)), 'lbx': -np.inf, 'ubx': np.inf, 'lbg': 0, 'ubg': 0, 'p': np.concatenate(([x0], path_msg), axis=0)}
        res = solver(**args)
        u_opt = res['x'].full().reshape((N_horizon, 2))
        
        # 发布Twist控制指令
        cmd_vel = Twist()
        cmd_vel.linear.x = u_opt[0, 0]  
        cmd_vel.angular.z = u_opt[0, 1]
        cmd_vel_pub.publish(cmd_vel)
        
    rate.sleep()