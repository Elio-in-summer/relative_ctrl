#include <use-ikfom-uav.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <random>
#include <algorithm> 
#include <Python.h>

Eigen::Vector3d position_measure;
Eigen::Quaterniond orientation_measure;
Eigen::Vector3d velocity_measure;
double measure_time;
Eigen::Vector3d acc_imu;
Eigen::Vector3d gyro_imu;
double imu_time;

void imu_cb(const sensor_msgs::Imu::ConstPtr &msg)
{
    acc_imu << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
    gyro_imu << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    imu_time = msg->header.stamp.toSec();
}

void odom_cb(const nav_msgs::Odometry::ConstPtr &msg)
{
    position_measure << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
    orientation_measure = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    velocity_measure << msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z;
    measure_time = msg->header.stamp.toSec();
}

int main(int argc, char **argv)
{
    // ros init
    ros::init(argc, argv, "ikfom_uav", ros::init_options::AnonymousName);
    ros::NodeHandle nh;

    ros::Subscriber imu_sub = nh.subscribe("/imu", 1, imu_cb);
    ros::Subscriber odom_sub = nh.subscribe("/odom", 1, odom_cb);

    ros::Publisher odom_ekf_pub = nh.advertise<nav_msgs::Odometry>("/odom_ekf", 1);
    ros::Publisher bias_acc_pub = nh.advertise<geometry_msgs::Vector3>("/bias_acc", 1);
    ros::Publisher bias_gyro_pub = nh.advertise<geometry_msgs::Vector3>("/bias_gyro", 1);
    ros::Publisher gravity_pub = nh.advertise<geometry_msgs::Vector3>("/gravity", 1);
    
    
    ros::Rate loop(50);
    state_ikfom init_state;
    esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF>::cov init_P;
    init_state.pos << pos_measure[0], pos_measure[1], pos_measure[2];
    init_state.rot = orientation_measure;
    init_state.vel << velocity_measure[0], velocity_measure[1], velocity_measure[2];
    init_state.bg << 0, 0, 0;
    init_state.ba << 0, 0, 0;
    init_state.grav = S2(0, 0, 1);

    init_P.setIdentity();
    init_P *= 1e6;
    esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF> kf(init_state,init_P);
    
    double epsi[state_ikfom::DOF] = {0.001};
    std::fill(epsi, epsi+state_ikfom::DOF, 0.001); // if the absolute of innovation of ekf update is smaller than epso, the update iteration is converged
    
    int Maximum_iter = 1e5;
    kf.init(get_f, df_dx, df_dw, h, dh_dx, dh_dv, Maximum_iter, epsi);


    /**************************** main ctrl loop *******************************/
    while (ros::ok())
    {   
        ros::spinOnce();

        // simulate the input
        Eigen::Matrix<double, process_noise_ikfom::DOF, process_noise_ikfom::DOF> Q;
        Q = Eigen::Matrix<double, process_noise_ikfom::DOF, process_noise_ikfom::DOF>::Identity() * 0.1; // process noise covariance: Q, an Eigen matrix
        input_ikfom in;
        // Add guassian noise to the input with covariance Q
        Eigen::VectorXd noise = process_noise_cov();
        in.acc = acc_imu;
        in.gyro = gyro_imu;
        std::cout << "noise: " << noise << std::endl;
        kf.predict(dt, Q, in); // TODO: dt is not defined

        Eigen::Matrix<double, measurement_ikfom::DOF, measurement_ikfom::DOF> R;
        R = Eigen::Matrix<double, measurement_ikfom::DOF, measurement_ikfom::DOF>::Identity() * 0.01; // measurement noise covariance: R, an Eigen matrix
        Eigen::VectorXd noise_z = R.diagonal().cwiseSqrt().asDiagonal() * Eigen::VectorXd::Random(measurement_ikfom::DOF);
        std::cout << "noise_z: " << noise_z << std::endl;

        measurement_ikfom z;
        z.pos_h = position_measure;
        z.rot_h = orientation_measure;
        kf.update_iterated(z, R); // measurement noise covariance: R, an Eigen matrix

        state_ikfom state_ekf = kf.get_x();
        nav_msgs::Odometry odom_ekf;
        odom_ekf.header.stamp = ros::Time::now();
        odom_ekf.header.frame_id = "world";
        odom_ekf.pose.pose.position.x = state_ekf.pos[0];
        odom_ekf.pose.pose.position.y = state_ekf.pos[1];
        odom_ekf.pose.pose.position.z = state_ekf.pos[2];
        odom_ekf.pose.pose.orientation.w = state_ekf.rot.w();
        odom_ekf.pose.pose.orientation.x = state_ekf.rot.x();
        odom_ekf.pose.pose.orientation.y = state_ekf.rot.y();
        odom_ekf.pose.pose.orientation.z = state_ekf.rot.z();
        odom_ekf.twist.twist.linear.x = state_ekf.vel[0];
        odom_ekf.twist.twist.linear.y = state_ekf.vel[1];
        odom_ekf.twist.twist.linear.z = state_ekf.vel[2];
        odom_ekf_pub.publish(odom_ekf);

        geometry_msgs::Vector3 bias_acc;
        bias_acc.x = state_ekf.ba[0];
        bias_acc.y = state_ekf.ba[1];
        bias_acc.z = state_ekf.ba[2];

        geometry_msgs::Vector3 bias_gyro;
        bias_gyro.x = state_ekf.bg[0];
        bias_gyro.y = state_ekf.bg[1];
        bias_gyro.z = state_ekf.bg[2];

        geometry_msgs::Vector3 gravity;
        gravity.x = state_ekf.grav[0];
        gravity.y = state_ekf.grav[1];
        gravity.z = state_ekf.grav[2];

        loop.sleep();
        
    }

    return 0;
}