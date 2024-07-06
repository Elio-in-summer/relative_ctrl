// This is a modification of the algorithm described in the following paper:
//W.  Xu  and  F.  Zhang. Fast-lio:  A  fast,  robust  lidar-inertial  odome-try  package  by  tightly-coupled  iterated  kalman  filter. 
//arXiv  preprintarXiv:2010.08196, 2020

#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include <chrono>

#include <use-ikfom-uav.hpp>
#include <so3_math.h>

/// *************Preconfiguration

#define MAX_INI_COUNT 500
#define G_m_s2 9.8099
#define DIM_OF_PROC_N 12

struct MeasureGroup {
    std::deque<sensor_msgs::Imu::ConstPtr> imu;
    nav_msgs::Odometry::ConstPtr odom;
    double last_odom_time;
};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  Eigen::Matrix<double, 12, 12> Q = process_noise_cov();
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF> &kf_state);

  void Reset();
  
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF> &kf_state, int &N);

  void ForwardPredict(const MeasureGroup &meas, esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF> &kf_state);
  // Eigen::Matrix3d Exp(const Eigen::Vector3d &ang_vel, const double &dt);

  void IntegrateGyr(const std::vector<sensor_msgs::Imu::ConstPtr> &v_imu);

  ros::NodeHandle nh;

  void Integrate(const sensor_msgs::ImuConstPtr &imu);
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);

  double scale_gravity;

  Eigen::Vector3d angvel_last;
  Eigen::Vector3d acc_s_last;

  Eigen::Matrix<double,DIM_OF_PROC_N,1> cov_proc_noise;

  Eigen::Vector3d cov_acc;
  Eigen::Vector3d cov_gyr;

  std::ofstream fout;

 private:
  /*** Whether is the first frame, init for first frame ***/
  bool b_first_frame_ = true;
  bool imu_need_init_ = true;

  int init_iter_num = 1;
  Eigen::Vector3d mean_acc;
  Eigen::Vector3d mean_gyr;

  // For timestamp usage
  sensor_msgs::ImuConstPtr last_imu_;

  /*** For gyroscope integration ***/
  double start_timestamp_;
  /// Making sure the equal size: v_imu_ and v_rot_
  std::deque<sensor_msgs::ImuConstPtr> v_imu_;
  std::vector<Eigen::Matrix3d> v_rot_pcl_;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), last_imu_(nullptr), start_timestamp_(-1)
{
  Eigen::Quaterniond q(0, 1, 0, 0);
  Eigen::Vector3d t(0, 0, 0);
  init_iter_num = 1;
  scale_gravity = 1.0;
  cov_acc       = Eigen::Vector3d(0.1, 0.1, 0.1);
  cov_gyr       = Eigen::Vector3d(0.1, 0.1, 0.1);
  mean_acc      = Eigen::Vector3d(0, 0, -1.0);
  mean_gyr      = Eigen::Vector3d(0, 0, 0);
  angvel_last   = Eigen::Vector3d(0, 0, 0);
  cov_proc_noise = Eigen::Matrix<double,DIM_OF_PROC_N,1>::Zero();
}

ImuProcess::~ImuProcess() {fout.close();}

void ImuProcess::Reset() 
{
  ROS_WARN("Reset ImuProcess");
  scale_gravity  = 1.0;
  angvel_last   = Eigen::Vector3d(0, 0, 0);
  cov_proc_noise = Eigen::Matrix<double,DIM_OF_PROC_N,1>::Zero();

  cov_acc   = Eigen::Vector3d(0.1, 0.1, 0.1);
  cov_gyr   = Eigen::Vector3d(0.1, 0.1, 0.1);
  mean_acc  = Eigen::Vector3d(0, 0, -1.0);
  mean_gyr  = Eigen::Vector3d(0, 0, 0);

  imu_need_init_ = true;
  b_first_frame_ = true;
  init_iter_num  = 1;

  last_imu_      = nullptr;

  //gyr_int_.Reset(-1, nullptr);
  start_timestamp_ = -1;
  v_imu_.clear();
  fout.close();

}


void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);

  Eigen::Vector3d cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    scale_gravity += (cur_acc.norm() - scale_gravity) / N;
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);


    N ++;
  }
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(-mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  kf_state.change_x(init_state);

  // esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF>::cov init_P = Eigen::Matrix<double, 23, 23>::Identity() * 0.001;
  // kf_state.change_P(init_P);
}


void ImuProcess::ForwardPredict(const MeasureGroup &meas, esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF> &kf_state)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  // const double &imu_beg_time = v_imu.front()->header.stamp.toSec(); 
  // const double &imu_end_time = v_imu.back()->header.stamp.toSec(); 
  const double &odom_time = meas.odom->header.stamp.toSec();
  const double &last_o_time = meas.last_odom_time;
  //* now the meas contains along the time line: imu_last, imu_1, imu_2, ..., imu_n, odom
  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  // esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF>::cov check_P = kf_state.get_P();

  /*** forward propagation at each imu point ***/
  Eigen::Vector3d angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  Eigen::Matrix3d R_imu;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    if (tail->header.stamp.toSec() < last_o_time)    continue; 
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    if(head->header.stamp.toSec() < last_o_time) 
    {
      dt = tail->header.stamp.toSec() - last_o_time; 
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr * 10000;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc * 10000;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    // double &&offs_t = tail->header.stamp.toSec() - last_o_time; 
  }

  auto &&tail = v_imu.back();
  dt = odom_time - tail->header.stamp.toSec();
  kf_state.predict(dt, Q, in);
  imu_state = kf_state.get_x();

}


void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF> &kf_state)
{
  auto t1 = std::chrono::high_resolution_clock::now();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.odom != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
      // cout<<"mean acc: "<<mean_acc<<" acc measures in word frame:"<<state.rot_end.transpose()*mean_acc<<endl;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
               imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
    }
    return;
  }

  /// Undistort points： the first point is assummed as the base frame
  /// Compensate lidar points with IMU rotation (with only rotation now)
  ForwardPredict(meas, kf_state);

  /// Record last measurements
  last_imu_   = meas.imu.back();

  auto t3 = std::chrono::high_resolution_clock::now();
  std::cout<<"[ IMU Process ]: Time: "<<std::chrono::duration_cast<std::chrono::microseconds>(t3 - t1).count() / 1000.0<<" ms"<<std::endl;
}