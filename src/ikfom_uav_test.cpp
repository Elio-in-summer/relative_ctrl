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

#define MAX_INI_COUNT 500
#define G_m_s2 9.8099
// 自定义结构体，用于存储测量数据
struct MeasureGroup {
    std::deque<sensor_msgs::Imu::Ptr> imu;
    nav_msgs::Odometry::ConstPtr odom;
};

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

class ImuProcess {
public:
    void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
    {   
        // ! Initialize some params through imu information
        // ! Need uav static on the ground
        ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);

        Eigen::Vector3d cur_acc, cur_gyr;

        for (const auto &imu : meas.imu)
        {
            const auto &imu_acc = imu->linear_acceleration;
            const auto &gyr_acc = imu->angular_velocity;
            cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
            cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

            mean_acc += (cur_acc - mean_acc) / N;
            mean_gyr += (cur_gyr - mean_gyr) / N;

            cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
            cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

            N++;
        }
        state_ikfom init_state = kf_state.get_x();
        init_state.grav = S2(-mean_acc / mean_acc.norm() * G_m_s2);
        init_state.bg = mean_gyr;
        kf_state.change_x(init_state);
    }



    void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
    {
        if (meas.imu.empty()) { return; }

        if (imu_need_init_)
        {
            int init_iter_num = 1;
            IMU_init(meas, kf_state, init_iter_num);
            last_imu_ = meas.imu.back();
            if (init_iter_num > MAX_INI_COUNT)
            {
                cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
                imu_need_init_ = false;
                ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; acc covariance: %.8f %.8f %.8f; gyr covariance: %.8f %.8f %.8f",
                    kf_state.get_x().grav[0], kf_state.get_x().grav[1], kf_state.get_x().grav[2], mean_acc.norm(), cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
            }
            return;
        }

        // Predict with IMU data
        for (const auto &imu : meas.imu)
        {
            double dt = (imu->header.stamp.toSec() - last_imu_->header.stamp.toSec());
            input_ikfom in;
            in.acc << imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z;
            in.gyro << imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z;
            Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
            Q.block<3, 3>(0, 0).diagonal() = cov_gyr * 10000;
            Q.block<3, 3>(3, 3).diagonal() = cov_acc * 10000;
            kf_state.predict(dt, Q, in);
            last_imu_ = imu;
        }

        // Update with Odom data
        measurement_ikfom z;
        z.pos_h << meas.odom->pose.pose.position.x, meas.odom->pose.pose.position.y, meas.odom->pose.pose.position.z;
        z.rot_h = Eigen::Quaterniond(meas.odom->pose.pose.orientation.w, meas.odom->pose.pose.orientation.x, meas.odom->pose.pose.orientation.y, meas.odom->pose.pose.orientation.z).toRotationMatrix();
        Eigen::Matrix<double, measurement_ikfom::DOF, measurement_ikfom::DOF> R;
        R.setIdentity();
        R *= 0.01; // measurement noise covariance
        kf_state.update_iterated(z, R);
    }

};




int main(int argc, char **argv)
{
    ros::init(argc, argv, "ikfom_uav_test", ros::init_options::AnonymousName);
    ros::NodeHandle nh;

    ros::Subscriber imu_sub = nh.subscribe("/imu", 1, imu_cb);
    ros::Subscriber odom_sub = nh.subscribe("/odom", 1, odom_cb);

    MeasureGroup Measures;
    ros::Rate loop(100);
    int count = 0;
    while (ros::ok())
    {
        ros::spinOnce();

        if (sync_packages(Measures))
        {   
            std::cout << "Measurements synced" << count << std::endl;
            // print Measures data's timestamp     
            for (auto imu : Measures.imu)
            {
                std::cout << "imu timestamps: " << imu->header.stamp << std::endl;
            }
            std::cout << "odom timestamp: " << Measures.odom->header.stamp << std::endl;
            count++;
        }

        loop.sleep();
    }

    return 0;
}
