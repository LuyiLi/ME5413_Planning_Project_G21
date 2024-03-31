#!/usr/bin/env python3

import casadi as ca
import numpy as np
import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, Quaternion
from geometry_msgs.msg import PoseStamped
import tf
from dynamic_reconfigure.msg import Config

# Initialize ROS node
rospy.init_node('mpc_node')

# MPC parameters
N = 20  # Prediction horizon
start_n = 11  # Starting point index

velocity = 1.0  # Velocity
omega_ref = 0.0  # Angular velocity

dt = 0.1  # Time step of the MPC


def callback(config):
    """
    Callback function for handling speed configuration updates.

    Args:
        config: The updated configuration object.

    Returns:
        None
    """
    global velocity
    # Set velocity to speed_target
    velocity = config.doubles[0].value

cmd_sub = rospy.Subscriber("/me5413_world/path_publisher_node/parameter_updates", Config, callback)


def mpc(p_ref, x0=[0, 0, 0]):
    """
    Model Predictive Control (MPC) algorithm for trajectory tracking.

    Args:
        p_ref (numpy.ndarray): Reference trajectory points.
        x0 (list, optional): Initial state. Defaults to [0, 0, 0].

    Returns:
        numpy.ndarray: Optimal control input.

    """
    # Define the system dynamics model
    x = ca.SX.sym('x', 3)  # State variables: position (x, y) and yaw angle
    u = ca.SX.sym('u', 2)  # Control variables: forward velocity v and angular velocity omega
    v = u[0]
    omega = u[1]

    xdot = ca.vertcat(v * ca.cos(x[2]),
                    v * ca.sin(x[2]), 
                    omega)  

    f = ca.Function('f', [x, u], [x + xdot * dt])


    # Define the MPC problem
    opti = ca.Opti()
    
    Q_np = np.diag([1, 1, 0.5])   # State weight (Numpy array)
    R_np = np.diag([0.25, 0.05])    # Control weight (Numpy array)

    Q = ca.MX(Q_np)  # Convert Numpy array to CasADi SX type
    R = ca.MX(R_np)

    # Define optimization variables
    X = opti.variable(N+1, 3)  # State sequence
    U = opti.variable(N, 2)    # Control sequence
    u_ref = np.array([velocity, omega_ref])

    # Define the objective function
    obj = 0
    for i in range(N):
        state_err = X[i+1, :].T - p_ref[i, :]
        state_err[2] = ca.if_else(state_err[2] > ca.pi, state_err[2] - 2*ca.pi,
                               ca.if_else(state_err[2] < -ca.pi, state_err[2] + 2*ca.pi, state_err[2]))
        
        obj += ca.mtimes([state_err.T, Q, state_err]) * N / (i + N)  # Control cost

        vel_err = U[i, :].T - u_ref

        obj += ca.mtimes([vel_err.T, R, vel_err]) * N / (i + N)


    opti.minimize(obj)

    # Add constraints
    for i in range(N):
        opti.subject_to(X[i+1, :].T == f(X[i, :], U[i, :]))  # Dynamics constraint

    # Set initial state
    opti.subject_to(X[0, :].T == x0)

    # Set control input constraints
    opti.subject_to(opti.bounded(0, U[:, 0], 1.5))  # Forward velocity limit
    opti.subject_to(opti.bounded(-2.5, U[:, 1], 2.5))  # Angular velocity limit

    # Solver options
    opts = {'ipopt.print_level': 0, 'print_time': 0}

    # Solve the optimization problem using IPOPT solver
    opti.solver('ipopt', opts)

    # Update reference path (based on actual situation)
        
    # Set reference path parameters
    opti.set_value(opti.parameter(N, 3), p_ref[:, :])

    # Solve the optimization problem
    sol = opti.solve()

    # Extract optimal control input
    u_opt = sol.value(U[0, :])
    mpc_pub(sol.value(X))
    # print(sol.value(obj))
    return u_opt

trajectory_publisher = rospy.Publisher('/mpc_trajectory', Path, queue_size=10)

def mpc_pub(mpc_traj):
    """
    Publishes the given MPC trajectory as a Path message.

    Args:
        mpc_traj (numpy.ndarray): The MPC trajectory to be published.

    Returns:
        None
    """
    # Create Path message
    path_msg1 = Path()
    path_msg1.header.stamp = rospy.Time.now()  # Set timestamp
    path_msg1.header.frame_id = 'world'  # Set frame

    # Add each trajectory point to Path message
    for i in range(mpc_traj.shape[0]):
        pose = PoseStamped()
        pose.header.frame_id = 'world'  # Set frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = mpc_traj[i][0]  # Set x coordinate
        pose.pose.position.y = mpc_traj[i][1]  # Set y coordinate
        pose.pose.position.z = 0.0  # Assume z coordinate is 0
        path_msg1.poses.append(pose)
    trajectory_publisher.publish(path_msg1)


# Subscribe to path topic
path_msg = None
p_ref = None
def path_callback(msg):
    """
    Callback function for processing path messages.

    Args:
        msg: The path message containing poses.

    Returns:
        None
    """
    global path_msg
    path_length = max(len(msg.poses), 60)
    path_msg = np.zeros((path_length, 3))

    j = 0
    p_len = 0
    domega = 0
    for i, pose in enumerate(msg.poses):
        path_msg[i, 0] = pose.pose.position.x
        path_msg[i, 1] = pose.pose.position.y
        quat = pose.pose.orientation
        path_msg[i, 2] = np.arctan2(2*(quat.w*quat.z + quat.x*quat.y), 
                                    1 - 2*(quat.y**2 + quat.z**2))   
        j = i
        # Add the length of the path
        if i > 10 and i <= 15:
            p_len += np.sqrt((path_msg[i, 0] - path_msg[i-1, 0])**2 + (path_msg[i, 1] - path_msg[i-1, 1])**2)

        # Add the angle difference
        if i > 10 and i <= 13:
            if path_msg[i, 2] - path_msg[i-1, 2] > np.pi:
                domega += path_msg[i, 2] - path_msg[i-1, 2] - 2*np.pi
            elif path_msg[i, 2] - path_msg[i-1, 2] < -np.pi:
                domega += path_msg[i, 2] - path_msg[i-1, 2] + 2*np.pi
            else:
                domega += path_msg[i, 2] - path_msg[i-1, 2]
            

    for i in range(j+1, path_length):
        path_msg[i, :] = path_msg[j, :]

    
    global p_ref
    p_ref = path_msg[start_n:start_n+N, :]

    global dt
    if j > start_n - 1:
        dt = p_len / min(j-10, 5) / velocity

    global omega_ref
    if j > start_n:
        omega_ref = domega / min(j-10, 3) / dt


path_sub = rospy.Subscriber('/me5413_world/planning/local_path', Path, path_callback)

cur_pose = None
def odom_callback(msg):
    """
    Callback function for the odometry message.

    Args:
        msg (Odometry): The odometry message containing the pose information.

    Returns:
        None
    """
    global cur_pose
    pos = msg.pose.pose.position
    quat = msg.pose.pose.orientation
    cur_pose = np.array([pos.x, pos.y, tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]])


odom_sub = rospy.Subscriber('/gazebo/ground_truth/state', Odometry, odom_callback)

# Control command publisher
cmd_vel_pub = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=1)

# Main MPC loop
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    if p_ref is not None and cur_pose is not None:
        u_opt = mpc(p_ref, cur_pose)
        # Publish Twist control command
        cmd_vel = Twist()
        cmd_vel.linear.x = u_opt[0]
        cmd_vel.angular.z = u_opt[1]
        cmd_vel_pub.publish(cmd_vel)
    rate.sleep()