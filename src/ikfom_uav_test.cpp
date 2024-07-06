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
#include <deque>
#include <mutex>
#include <condition_variable>

#include <IMU_Processing.hpp>

// 互斥量和条件变量
std::mutex mtx_buffer;
std::condition_variable sig_buffer;
std::deque<sensor_msgs::Imu::Ptr> imu_buffer;
std::deque<nav_msgs::Odometry::ConstPtr> odom_buffer;
double last_timestamp_imu = 0;
bool flg_reset = false;
bool imu_need_init_ = true;
Eigen::Vector3d mean_acc(0, 0, 0), mean_gyr(0, 0, 0);
Eigen::Vector3d cov_acc(0, 0, 0), cov_gyr(0, 0, 0);
sensor_msgs::Imu::Ptr last_imu_;

void imu_cb(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    double timestamp = msg->header.stamp.toSec();

    std::lock_guard<std::mutex> lock(mtx_buffer);
    if (timestamp < last_timestamp_imu)
    {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
        flg_reset = true;
    }
    last_timestamp_imu = timestamp;
    imu_buffer.push_back(msg);
    sig_buffer.notify_all();
}

void odom_cb(const nav_msgs::Odometry::ConstPtr &msg_in)
{
    std::lock_guard<std::mutex> lock(mtx_buffer);
    odom_buffer.push_back(msg_in);
    sig_buffer.notify_all();
}

bool sync_packages(MeasureGroup &meas)
{
    std::lock_guard<std::mutex> lock(mtx_buffer);

    if (odom_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    meas.odom = odom_buffer.front();
    double odom_time = meas.odom->header.stamp.toSec();

    if (last_timestamp_imu < odom_time) {
        return false;
    }

    meas.imu.clear();
    while (!imu_buffer.empty() && imu_buffer.front()->header.stamp.toSec() <= odom_time) {
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    odom_buffer.pop_front();
    return true;
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "ikfom_uav_test", ros::init_options::AnonymousName);
    ros::NodeHandle nh;

    ros::Subscriber imu_sub = nh.subscribe("/imu", 1, imu_cb);
    ros::Subscriber odom_sub = nh.subscribe("/odom", 1, odom_cb);

    ros::Publisher odom_ekf_pub = nh.advertise<nav_msgs::Odometry>("/odom_ekf", 1);
    ros::Publisher bias_acc_pub = nh.advertise<geometry_msgs::Vector3>("/bias_acc", 1);
    ros::Publisher bias_gyro_pub = nh.advertise<geometry_msgs::Vector3>("/bias_gyro", 1);
    ros::Publisher gravity_pub = nh.advertise<geometry_msgs::Vector3>("/gravity", 1);
    
    state_ikfom init_state;
    esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF>::cov init_P;
    esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF> kf(init_state, init_P);

    double epsi[state_ikfom::DOF] = {0.001};
    std::fill(epsi, epsi + state_ikfom::DOF, 0.001);

    int Maximum_iter = 1e5;
    kf.init(get_f, df_dx, df_dw, h, dh_dx, dh_dv, Maximum_iter, epsi);

    MeasureGroup Measures;
    ImuProcess p_imu;

    ros::Rate loop(100);
    int count = 0;
    while (ros::ok())
    {
        ros::spinOnce();

        if (sync_packages(Measures))
        {   
            std::cout << "\033[31mMeasurements synced\033[0m" << count << std::endl;
            // print Measures data's timestamp     
            for (auto imu : Measures.imu)
            {
                std::cout << "imu timestamps: " << imu->header.stamp << std::endl;
            }
            std::cout << "odom timestamp: " << Measures.odom->header.stamp << std::endl;
            count++;
            p_imu.Process(Measures, kf);
            measurement_ikfom z;
            z.pos_h << Measures.odom->pose.pose.position.x, Measures.odom->pose.pose.position.y, Measures.odom->pose.pose.position.z;
            z.rot_h = Eigen::Quaterniond(Measures.odom->pose.pose.orientation.w, Measures.odom->pose.pose.orientation.x, Measures.odom->pose.pose.orientation.y, Measures.odom->pose.pose.orientation.z).toRotationMatrix();
            Eigen::Matrix<double, measurement_ikfom::DOF, measurement_ikfom::DOF> R;
            R.setIdentity();
            R *= 0.01; // measurement noise covariance
            kf.update_iterated(z, R);

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
            bias_acc_pub.publish(bias_acc);

            geometry_msgs::Vector3 bias_gyro;
            bias_gyro.x = state_ekf.bg[0];
            bias_gyro.y = state_ekf.bg[1];
            bias_gyro.z = state_ekf.bg[2];
            bias_gyro_pub.publish(bias_gyro);

            geometry_msgs::Vector3 gravity;
            gravity.x = state_ekf.grav[0];
            gravity.y = state_ekf.grav[1];
            gravity.z = state_ekf.grav[2];
            gravity_pub.publish(gravity);
        }

        loop.sleep();
    }

    return 0;
}
