#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <gazebo_msgs/ModelStates.h>
#include <Eigen/Eigen>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

#define TRANS_LOOPRATE 100
Eigen::Vector3d W_p_B;
Eigen::Quaterniond W_q_B;
Eigen::Vector3d W_p_E;
Eigen::Quaterniond W_q_E;

Eigen::Vector3d B_p_E;
Eigen::Quaterniond B_q_E;


bool get_model_state = false;

void gazebo_cb(const gazebo_msgs::ModelStates::ConstPtr &msg)
{
    // the target's model name is "tag36h11_0", the uav's model name is "iris_0"
    // get the pose of the uav and the target
    for (int i = 0; i < msg->name.size(); i++)
    {
        if (msg->name[i] == "iris_0")
        {
            W_p_B.x() = msg->pose[i].position.x;
            W_p_B.y() = msg->pose[i].position.y;
            W_p_B.z() = msg->pose[i].position.z;
            W_q_B.x() = msg->pose[i].orientation.x;
            W_q_B.y() = msg->pose[i].orientation.y;
            W_q_B.z() = msg->pose[i].orientation.z;
            W_q_B.w() = msg->pose[i].orientation.w;
            // std::cout   << "W_p_B: " << W_p_B.transpose() << std::endl;
            // std::cout   << "W_q_B: " << W_q_B.x() << " " << W_q_B.y() << " " << W_q_B.z() << " " << W_q_B.w() << std::endl;
        }
        else if (msg->name[i] == "tag36h11_0")
        {
            W_p_E.x() = msg->pose[i].position.x;
            W_p_E.y() = msg->pose[i].position.y;
            W_p_E.z() = msg->pose[i].position.z;
            W_q_E.x() = msg->pose[i].orientation.x;
            W_q_E.y() = msg->pose[i].orientation.y;
            W_q_E.z() = msg->pose[i].orientation.z;
            W_q_E.w() = msg->pose[i].orientation.w;
            // std::cout   << "W_p_E: " << W_p_E.transpose() << std::endl;
            // std::cout   << "W_q_E: " << W_q_E.x() << " " << W_q_E.y() << " " << W_q_E.z() << " " << W_q_E.w() << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
    // ros init
    ros::init(argc, argv, "trans_relative_pose", ros::init_options::AnonymousName);
    ros::NodeHandle nh;
    // by this node, we get the relative pose of the uav and the target which is expressed in the uav frame
    // B is uav's frame and E is target's frame, so we calculate B_p_E and B_q_E, and publish them

    // ros pub and sub
    // we need to subscribe the pose of uav and the target from gazebo
    // the target's model name is "tag36h11_0", the uav's model name is "iris_0"
    ros::Publisher relative_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/relative_pose", 1);   
    ros::Subscriber uav_pose_sub = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1, gazebo_cb);
    
    ros::Rate loop(TRANS_LOOPRATE);


    /**************************** main ctrl loop *******************************/
    while (ros::ok())
    {
        ros::spinOnce();
        B_q_E = W_q_B.inverse() * W_q_E;
        B_p_E = W_q_B.inverse() * (W_p_E - W_p_B);
        std::cout   << "B_p_E: " << B_p_E.transpose() << std::endl;
        std::cout   << "B_q_E: " << B_q_E.x() << " " << B_q_E.y() << " " << B_q_E.z() << " " << B_q_E.w() << std::endl;
        geometry_msgs::PoseStamped relative_pose;
        relative_pose.header.stamp = ros::Time::now();
        relative_pose.pose.position.x = B_p_E(0);
        relative_pose.pose.position.y = B_p_E(1);
        relative_pose.pose.position.z = B_p_E(2);
        relative_pose.pose.orientation.x = B_q_E.x();
        relative_pose.pose.orientation.y = B_q_E.y();
        relative_pose.pose.orientation.z = B_q_E.z();
        relative_pose.pose.orientation.w = B_q_E.w();
        relative_pose_pub.publish(relative_pose);
        loop.sleep();
    }

    return 0;
}