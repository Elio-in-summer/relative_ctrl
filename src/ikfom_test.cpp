#include <use-ikfom-test.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <geometry_msgs/Vector3.h>
#include <random>
#include <algorithm> 
#include <Python.h>

int main(int argc, char **argv)
{
    // ros init
    ros::init(argc, argv, "ikfom_test", ros::init_options::AnonymousName);
    ros::NodeHandle nh;
    ros::Publisher pos_pub = nh.advertise<geometry_msgs::Vector3>("/pos_simu", 1);   
    ros::Publisher vel_pub = nh.advertise<geometry_msgs::Vector3>("/vel_simu", 1);
    ros::Publisher acc_pub = nh.advertise<geometry_msgs::Vector3>("/acc_simu", 1);
    ros::Publisher pos_ekf_pub = nh.advertise<geometry_msgs::Vector3>("/pos_ekf", 1);
    ros::Publisher vel_ekf_pub = nh.advertise<geometry_msgs::Vector3>("/vel_ekf", 1);
    ros::Publisher pos_measure_pub = nh.advertise<geometry_msgs::Vector3>("/pos_measure", 1);
    ros::Publisher acc_in_pub = nh.advertise<geometry_msgs::Vector3>("/acc_in", 1);
    
    ros::Rate loop(25);
    state_ikfom init_state;
    esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF>::cov init_P;
    init_state.pos << 0, 0, 0;
    init_state.vel << 0, 0, 0;
    init_P.setIdentity();
    init_P *= 1e6;
    esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF> kf(init_state,init_P);
    
    double epsi[state_ikfom::DOF] = {0.001};
    std::fill(epsi, epsi+state_ikfom::DOF, 0.001); // if the absolute of innovation of ekf update is smaller than epso, the update iteration is converged
    
    int Maximum_iter = 1e5;
    kf.init(get_f, df_dx, df_dw, h, dh_dx, dh_dv, Maximum_iter, epsi);

    /**************************** main ctrl loop *******************************/
    double time_last = ros::Time::now().toSec();
    double time_acuum = 0;
    while (ros::ok())
    {   
        ros::spinOnce();
        double time = ros::Time::now().toSec();
        double dt = time - time_last;
        time_last = time;
        std::cout << "dt: " << dt << std::endl;
        std::cout << "time_acuum: " << time_acuum << std::endl;
        if (dt < 0.03 or dt > 0.05)
        {
            loop.sleep();
            continue;
        }
        else
        {
            time_acuum += dt;
            // simulate the real state
            Eigen::Vector3d real_acc;
            real_acc << 1, 2, 3;
            Eigen::Vector3d real_vel =  real_acc * time_acuum;
            Eigen::Vector3d real_pos =  0.5 * real_acc * time_acuum * time_acuum;

            // simulate the input
            Eigen::Matrix<double, process_noise_ikfom::DOF, process_noise_ikfom::DOF> Q;
            Q = Eigen::Matrix<double, process_noise_ikfom::DOF, process_noise_ikfom::DOF>::Identity() * 0.1; // process noise covariance: Q, an Eigen matrix
            input_ikfom in;
            // Add guassian noise to the input with covariance Q
            Eigen::VectorXd noise = Q.diagonal().cwiseSqrt().asDiagonal() * Eigen::VectorXd::Random(process_noise_ikfom::DOF);
            in.acc = 2*noise + real_acc;
            std::cout << "noise: " << noise << std::endl;
            kf.predict(dt, Q, in); // process noise covariance: Q, an Eigen matrix

            // simulate the measurement, two order integration of the input
            Eigen::Vector3d pos_measure = real_pos;
            // Add guassian noise to the measurement with covariance R
            Eigen::Matrix<double, measurement_ikfom::DOF, measurement_ikfom::DOF> R;
            R = Eigen::Matrix<double, measurement_ikfom::DOF, measurement_ikfom::DOF>::Identity() * 0.1; // measurement noise covariance: R, an Eigen matrix
            Eigen::VectorXd noise_z = R.diagonal().cwiseSqrt().asDiagonal() * Eigen::VectorXd::Random(measurement_ikfom::DOF);
            std::cout << "noise_z: " << noise_z << std::endl;
            pos_measure = real_pos + noise_z;

            measurement_ikfom z;
            z.position = pos_measure;
            kf.update_iterated(z, R); // measurement noise covariance: R, an Eigen matrix

            geometry_msgs::Vector3 pos_msg;
            pos_msg.x = real_pos[0];
            pos_msg.y = real_pos[1];
            pos_msg.z = real_pos[2];
            pos_pub.publish(pos_msg);

            geometry_msgs::Vector3 vel_msg;
            vel_msg.x = real_vel[0];
            vel_msg.y = real_vel[1];
            vel_msg.z = real_vel[2];
            vel_pub.publish(vel_msg);

            geometry_msgs::Vector3 acc_msg;
            acc_msg.x = real_acc[0];
            acc_msg.y = real_acc[1];
            acc_msg.z = real_acc[2];
            acc_pub.publish(acc_msg);

            state_ikfom state_ekf = kf.get_x();
            geometry_msgs::Vector3 pos_ekf_msg;
            pos_ekf_msg.x = state_ekf.pos[0];
            pos_ekf_msg.y = state_ekf.pos[1];
            pos_ekf_msg.z = state_ekf.pos[2];
            pos_ekf_pub.publish(pos_ekf_msg);

            geometry_msgs::Vector3 vel_ekf_msg;
            vel_ekf_msg.x = state_ekf.vel[0];
            vel_ekf_msg.y = state_ekf.vel[1];
            vel_ekf_msg.z = state_ekf.vel[2];
            vel_ekf_pub.publish(vel_ekf_msg);

            geometry_msgs::Vector3 pos_measure_msg;
            pos_measure_msg.x = pos_measure[0];
            pos_measure_msg.y = pos_measure[1];
            pos_measure_msg.z = pos_measure[2];
            pos_measure_pub.publish(pos_measure_msg);

            geometry_msgs::Vector3 acc_in_msg;
            acc_in_msg.x = in.acc[0];
            acc_in_msg.y = in.acc[1];
            acc_in_msg.z = in.acc[2];
            acc_in_pub.publish(acc_in_msg);

            loop.sleep();
        }
        
    }

    return 0;
}