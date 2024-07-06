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

#ifndef USE_IKFOM_UAV_H
#define USE_IKFOM_UAV_H

#include <IKFoM_toolkit/esekfom/esekfom.hpp>

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98099, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos)) 
((SO3, rot))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((S2, grav))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_ikfom,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

MTK_BUILD_MANIFOLD(measurement_ikfom,
((vect3, pos_h))
((SO3, rot_h))
);

MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{	
	MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
	MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001);// 0.03
	MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na, 0.0001); // *dt 0.01 0.01 * dt * dt 0.05
	MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.003); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.003);   //0.001 0.05 0.0001/out 0.01
	return cov;
}


Eigen::Matrix<double, state_ikfom::DIM, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, state_ikfom::DIM, 1> res = Eigen::Matrix<double, state_ikfom::DIM, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg); // omega = gyro - bias
	vect3 a_inertial = s.rot * (in.acc-s.ba); 
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) =  omega[i]; 
		res(i + 6) = a_inertial[i] + s.grav[i]; 
	}
	return res;
}

Eigen::Matrix<double, state_ikfom::DIM, state_ikfom::DOF> df_dx(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, state_ikfom::DIM, state_ikfom::DOF> cov = Eigen::Matrix<double, state_ikfom::DIM, state_ikfom::DOF>::Zero();
	cov.template block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	cov.template block<3, 3>(6, 3) = -s.rot.toRotationMatrix()*MTK::hat(acc_);
	cov.template block<3, 3>(6, 12) = -s.rot.toRotationMatrix();
	Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	s.S2_Mx(grav_matrix, vec, 15); // here 15 is the beginning index of grav
	cov.template block<3, 2>(6, 15) =  grav_matrix; 
	cov.template block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity(); 
	return cov;
}


Eigen::Matrix<double, state_ikfom::DIM, process_noise_ikfom::DOF> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, state_ikfom::DIM, process_noise_ikfom::DOF> cov = Eigen::Matrix<double, state_ikfom::DIM, process_noise_ikfom::DOF>::Zero();
	cov.template block<3, 3>(6, 3) = -s.rot.toRotationMatrix();
	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();
	return cov;
}

measurement_ikfom h(state_ikfom &s, bool &valid) // the iteration stops before convergence whenever the user set valid as false
{
	if (false){ valid = false; } // other conditions could be used to stop the ekf update iteration before convergence, otherwise the iteration will not stop until the condition of convergence is satisfied.
	measurement_ikfom h_;
	h_.pos_h = s.pos;
	h_.rot_h = s.rot;
	return h_;
}
Eigen::Matrix<double, measurement_ikfom::DIM, state_ikfom::DOF> dh_dx(state_ikfom &s, bool &valid) 
{
	if (false){ valid = false; } // other conditions could be used to stop the ekf update iteration before convergence, otherwise the iteration will not stop until the condition of convergence is satisfied.
	Eigen::Matrix<double, measurement_ikfom::DIM, state_ikfom::DOF> cov = Eigen::Matrix<double, measurement_ikfom::DIM, state_ikfom::DOF>::Zero();
	// dh_dx = [I,0]
	cov.template block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
	return cov;
} 

Eigen::Matrix<double, measurement_ikfom::DIM, measurement_ikfom::DOF> dh_dv(state_ikfom &s, bool &valid) 
{
	if (false){ valid = false; } // other conditions could be used to stop the ekf update iteration before convergence, otherwise the iteration will not stop until the condition of convergence is satisfied.
	Eigen::Matrix<double, measurement_ikfom::DIM, measurement_ikfom::DOF> cov = Eigen::Matrix<double, measurement_ikfom::DIM, measurement_ikfom::DOF>::Zero();
	// dh_dv = -I
	cov.template block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(3, 3) = -Eigen::Matrix3d::Identity();
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