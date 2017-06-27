/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;

	normal_distribution<double> x_init(x, std[0]);
	normal_distribution<double> y_init(y, std[1]);
	normal_distribution<double> theta_init(theta, std[2]);

	num_particles = 100;

	for(int i = 0; i < num_particles; i++) {
		Particle p = Particle();
		p.id = i;
		p.x = x_init(gen);
		p.y = y_init(gen);
		p.theta = theta_init(gen);
		p.weight = 1.0;
		weights.push_back(1.0);
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	normal_distribution<double> x_noise(0, std_pos[0]);
	normal_distribution<double> y_noise(0, std_pos[1]);
	normal_distribution<double> theta_noise(0, std_pos[2]);

	double x_p, y_p, theta_t, theta_p, vyaw, vdt;

	for(int i = 0; i < particles.size(); i++) {
		theta_t = particles[i].theta;
		theta_p = theta_t + (yaw_rate * delta_t);
		
		if(fabs(yaw_rate) > 0.0001) {
			vyaw = velocity / yaw_rate;
			x_p = vyaw * (sin(theta_p) - sin(theta_t));
			y_p = vyaw * (cos(theta_t) - cos(theta_p));
		} else {
			vdt = velocity * delta_t;
			x_p = vdt * cos(theta_t);
			y_p = vdt * sin(theta_t);
		}

		particles[i].x += x_p + x_noise(gen);
		particles[i].y += y_p + y_noise(gen);
		particles[i].theta = theta_p + theta_noise(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double min_dist, dist, dx, dy;
	int min_i;

	for(int i_obs = 0; i_obs < observations.size(); i_obs++) {
		auto obs = observations[i_obs];
		min_dist = 1000000.0;

		for(int i = 0; i < predicted.size(); i++) {
			auto pred = predicted[i];
			dx = (obs.x - pred.x);
			dy = (obs.y - pred.y);
			dist = dx*dx + dy*dy;

			if(dist < min_dist) {
				min_dist = dist;
				min_i = i;
			}
		}

		observations[i_obs].id = min_i;
	} 
}

inline LandmarkObs car_to_map_coor(LandmarkObs observation, Particle p) {
	LandmarkObs res;

	res.x = p.x + observation.x * cos(p.theta) - observation.y * sin(p.theta);
	res.y = p.y + observation.x * sin(p.theta) + observation.y * cos(p.theta);
	res.id = observation.id;

	return res;
}

inline vector<LandmarkObs> car_to_map_coor_vec(vector<LandmarkObs> observations, Particle p) {
	vector<LandmarkObs> res;

	for(int i = 0; i < observations.size(); i++) {
		LandmarkObs obs = observations[i];
		res.push_back(car_to_map_coor(obs,p));
	}
	return res;
}

inline double gauss2d(LandmarkObs observation, LandmarkObs landmark, double std[]) {
	double sigma_x = std[0]*std[0];
	double sigma_y = std[1]*std[1];
	double denom = 2*M_PI*std[0]*std[1];
	double dx = observation.x - landmark.x;
	double dy = observation.y - landmark.y;
	double e = -0.5*(dx*dx/sigma_x + dy*dy/sigma_y);
	return exp(e)/denom;
}

inline double total_gauss2d(vector<LandmarkObs> observations, vector<LandmarkObs> landmarks, double std[]) {
	double res = 1.0;
	for(int i = 0; i < observations.size(); i++) {
		res *= gauss2d(observations[i], landmarks[i], std);
	}
	return res;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double sr2 = sensor_range * sensor_range;
	double px, py, lmx, lmy, dx, dy;

	for(int i = 0; i < particles.size(); i++) {
		Particle p = particles[i];
		vector<LandmarkObs> predicted_landmarks;
		vector<Map::single_landmark_s> map_landmarks_list = map_landmarks.landmark_list;
		
		px = p.x;
		py = p.y;

		for(int j = 0; j < map_landmarks_list.size(); j++) {
			auto landmark = map_landmarks_list[j];
			lmx = landmark.x_f;
			lmy = landmark.y_f;
			dx = px - lmx;
			dy = py - lmy;

			if((dx*dx + dy*dy) <= sr2) {
				LandmarkObs landmark_pred;
				landmark_pred.x = lmx;
				landmark_pred.y = lmy;
				landmark_pred.id = landmark.id_i;
				predicted_landmarks.push_back(landmark_pred);
			}
		}

		vector<LandmarkObs> observations_map_coor = car_to_map_coor_vec(observations, p);

		dataAssociation(predicted_landmarks, observations_map_coor);

		double proba = 1.0;

		for(int k = 0; k < observations_map_coor.size(); k++) {
			LandmarkObs observation = observations_map_coor[k];
			LandmarkObs landmark_pred = predicted_landmarks[observation.id];
			proba *= gauss2d(observation, landmark_pred, std_landmark);
		}

		particles[i].weight = proba;
		weights[i] = proba;

	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> d(weights.begin(), weights.end());
	vector<Particle> new_particles;

	for(int i = 0; i < particles.size(); i++) {
		new_particles.push_back(particles[d(gen)]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
