#ifndef LASERSCAN_SIMPLE
#define LASERSCAN_SIMPLE
#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "simple_laserscan/SimpleScan.h"
using namespace std;

class SimpleLaserScan
{
	private:
		/* ROS node */
		ros::NodeHandle nh_;
		/* ROS Subscriber and Publisher */
		ros::Subscriber scan_sub_;
		ros::Publisher simple_pub_;
		/* Number of directions for simplified scan */
		int dir_num_;
		/* Simplified laserscan data */
		vector<double> simple_data_;
		/* Node update rate */
		const int pub_rate_ = 20;
	public:
		/* Init function */
		SimpleLaserScan(ros::NodeHandle &n, int dir_num, std::string robot_name);
		/* Callback function for Laser Scan */
		void scanCallback(const sensor_msgs::LaserScan::ConstPtr &msg);
		/* Run function */
		void run(int time);
};
#endif
