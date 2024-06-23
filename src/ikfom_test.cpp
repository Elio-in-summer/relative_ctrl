#include <use-ikfom.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <geometry_msgs/Vector3.h>

int main(int argc, char **argv)
{
    // ros init
    ros::init(argc, argv, "trans_relative_pose", ros::init_options::AnonymousName);
    ros::NodeHandle nh;
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::Vector3>("/pos_simu", 1);   
    ros::Publisher vel_pub = nh.advertise<geometry_msgs::Vector3>("/vel_simu", 1);
    ros::Publisher acc_pub = nh.advertise<geometry_msgs::Vector3>("/acc_simu", 1);
    
    ros::Rate loop(25);
    state_ikfom init_state;
    esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF>::cov init_P;
    init_state.pos << 0, 0, 0;
    init_state.vel << 0, 0, 0;
    init_P.setIdentity();
    init_P *= 1e6;
    esekfom::esekf<state_ikfom, process_noise_ikfom::DOF, input_ikfom, measurement_ikfom, measurement_ikfom::DOF> kf(init_state,init_P);
    double epsi[state_dof] = {0.001};
    fill(epsi, epsi+state_dof, 0.001); // if the absolute of innovation of ekf update is smaller than epso, the update iteration is converged
    kf.init(f, df_dx, df_dw, h, dh_dx, dh_dv, Maximum_iter, epsi);



    /**************************** main ctrl loop *******************************/
    while (ros::ok())
    {
        ros::spinOnce();
        
        loop.sleep();
    }

    return 0;
}