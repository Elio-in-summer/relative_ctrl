/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef USE_IKFOM_H
#define USE_IKFOM_H

#include <IKFoM_toolkit/esekfom/esekfom.hpp>

// ! Select and instantiate the primitive manifolds:
typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98099, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

//! the state length is not necessary to be the same as the state_ikfom::DOF
//! state length is the length of f, which obeys x \oplus f rule
//! state dof is the length of \delta x, which obeys x \boxplus \delta x rule
//! for example, for S2, the state length is 3, but the state dof is 2
#define state_length 6

// ! Build system state, input and measurement 
// ! as compound manifolds which are composed of the primitive manifolds:
MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos))
((vect3, vel))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
);

MTK_BUILD_MANIFOLD(process_noise_ikfom,
((vect3, w_a))
);

MTK_BUILD_MANIFOLD(measurement_ikfom,
((vect3, position))
);


// ! Implement the vector field and its differentiation
//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
Eigen::Matrix<double, state_length, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, state_length, 1> res = Eigen::Matrix<double, state_length, 1>::Zero(); 
	res(0) = s.vel[0];
	res(1) = s.vel[1];
	res(2) = s.vel[2];
	res(3) = in.acc[0];
	res(4) = in.acc[1];
	res(5) = in.acc[2];
	return res;
}

Eigen::Matrix<double, state_length, state_ikfom::DOF> df_dx(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, state_length, state_ikfom::DOF> cov = Eigen::Matrix<double, state_length, state_ikfom::DOF>::Zero();
	// df_dx = [0,I;0,0]
	cov.template block<3, 3>(3, 0) = Eigen::Matrix3d::Identity(); 
	return cov;
}


Eigen::Matrix<double, state_length, process_noise_ikfom::DOF> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, state_length, process_noise_ikfom::DOF> cov = Eigen::Matrix<double, state_length, process_noise_ikfom::DOF>::Zero();
	// df_dw = [0;I]
	cov.template block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();
	return cov;
}

measurement_ikfom h_share(state_ikfom &s, esekfom::share_datastruct<state_ikfom, measurement_ikfom> &share_data) 
{
	if(share_data.converge) {} // this value is true means iteration is converged 
	if(false) share_data.valid = false; // the iteration stops before convergence when this value is false if other conditions are satified
	share_data.h_x.setZero(); // h_x is the result of the measurement function
	// h_x = [I,0]
	share_data.h_x.template block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
	share_data.h_v.setZero(); // h_v is the Jacobian of the measurement function
	// h_v = -I
	share_data.h_v.template block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
	share_data.R = Eigen::Matrix3d::Identity() * 0.1; // R is the covariance of the measurement noise
	// share_data.z = z; // z is the obtained measurement 

	measurement_ikfom h_;
	h_.position = s.pos;
	return h_;
}

measurement_ikfom h(state_ikfom &s, bool &valid) // the iteration stops before convergence whenever the user set valid as false
{
	if (false){ valid = false; } // other conditions could be used to stop the ekf update iteration before convergence, otherwise the iteration will not stop until the condition of convergence is satisfied.
	measurement_ikfom h_;
	h_.position = s.pos;
	return h_;
}

Eigen::Matrix<double, measurement_ikfom::DOF, state_ikfom::DOF> dh_dx(state_ikfom &s, bool &valid) 
{
	if (false){ valid = false; } // other conditions could be used to stop the ekf update iteration before convergence, otherwise the iteration will not stop until the condition of convergence is satisfied.
	Eigen::Matrix<double, measurement_ikfom::DOF, state_ikfom::DOF> cov = Eigen::Matrix<double, measurement_ikfom::DOF, state_ikfom::DOF>::Zero();
	// dh_dx = [I,0]
	cov.template block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
	return cov;
} 

Eigen::Matrix<double, measurement_ikfom::DOF, measurement_ikfom::DOF> dh_dv(state_ikfom &s, bool &valid) 
{
	if (false){ valid = false; } // other conditions could be used to stop the ekf update iteration before convergence, otherwise the iteration will not stop until the condition of convergence is satisfied.
	Eigen::Matrix<double, measurement_ikfom::DOF, measurement_ikfom::DOF> cov = Eigen::Matrix<double, measurement_ikfom::DOF, measurement_ikfom::DOF>::Zero();
	// dh_dv = -I
	cov.template block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
	return cov;
}

vect3 SO3ToEuler(const SO3 &orient) 
{
	Eigen::Matrix<double, 3, 1> _ang;
	Eigen::Vector4d q_data = orient.coeffs().transpose();
	//scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
	double sqw = q_data[3]*q_data[3];
	double sqx = q_data[0]*q_data[0];
	double sqy = q_data[1]*q_data[1];
	double sqz = q_data[2]*q_data[2];
	double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
	double test = q_data[3]*q_data[1] - q_data[2]*q_data[0];

	if (test > 0.49999*unit) { // singularity at north pole
	
		_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
	if (test < -0.49999*unit) { // singularity at south pole
		_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
		
	_ang <<
			std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
			std::asin (2*test/unit),
			std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	vect3 euler_ang(temp, 3);
		// euler_ang[0] = roll, euler_ang[1] = pitch, euler_ang[2] = yaw
	return euler_ang;
}

#endif