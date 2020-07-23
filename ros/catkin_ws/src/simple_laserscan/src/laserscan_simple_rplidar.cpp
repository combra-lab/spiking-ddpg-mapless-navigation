#include <laserscan_simple.h>

SimpleLaserScan::SimpleLaserScan(ros::NodeHandle &n, int dir_num, std::string robot_name)
{
	nh_ = n;
	simple_pub_ = nh_.advertise<simple_laserscan::SimpleScan>(robot_name + "/simplescan", 5);
	scan_sub_ = nh_.subscribe(robot_name + "/lpscan", 5, &SimpleLaserScan::scanCallback, this);
	dir_num_ = dir_num;
	simple_data_.resize(dir_num_);
	for(int num=0; num<dir_num_; num++)
	{
		simple_data_[num] = 0.0;
	}
	ROS_INFO("SIMPLE LASERSCAN NODE CREATED ...");
}

void SimpleLaserScan::scanCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
{
	vector<float> ranges = msg->ranges;
	int ranges_size = ranges.size();
	/* Change all nan values in ranges to zero */
	for(int num=0; num<ranges_size; num++)
	{
		if(isinf(ranges[num]) == true)
		{
			ranges[num] = 0.0;
		}
		else
		{
			continue;
		}
	}
	/* Find medians for each direction chunk */
	int dir_chunk_size = (int)((double)ranges_size / (double)dir_num_);
	int dir_chunk_start = (ranges_size - dir_chunk_size * dir_num_) / 2;
	int median_pt = dir_chunk_size / 2;
	for(int dir=0; dir<dir_num_; dir++)
	{
		int tmp_chunk_start = dir_chunk_start + dir * dir_chunk_size;
		int tmp_chunk_end = tmp_chunk_start + dir_chunk_size;
		int tmp_chunk_middle = tmp_chunk_start + dir_chunk_size / 2;
		nth_element(ranges.begin()+tmp_chunk_start, ranges.begin()+tmp_chunk_middle, ranges.begin()+tmp_chunk_end);
		simple_data_[dir] = (double)ranges[tmp_chunk_middle];
	}
}

void SimpleLaserScan::run(int time)
{
	ros::Rate loop_rate(pub_rate_);
	/* If time is -1 then always run */
	int loop_ita = 0;
	while(nh_.ok())
	{
		ros::spinOnce();
		/* Publish Simple Scan */
		simple_laserscan::SimpleScan tmp_msg;
		tmp_msg.stamp = ros::Time::now();
		tmp_msg.number = dir_num_;
		tmp_msg.data = simple_data_;
		simple_pub_.publish(tmp_msg);
		/* Check time */
		loop_ita++;
		if(time != -1 && time == loop_ita / pub_rate_)
		{
			ROS_INFO("LASER SCAN SIMPLIFIER END PUBLISHING ...");
			break;
		}
		loop_rate.sleep();
	}
}

/* Main Function */
int main(int argc, char** argv)
{
	ros::init(argc, argv, "simple_laserscan_rplidar", ros::init_options::AnonymousName);
	ros::NodeHandle n;
	std::string robot_name;
	if(n.getParam("multi_robot_name", robot_name))
	{
	    ROS_INFO("Got Param %s", robot_name.c_str());
	}
	else
	{
	    ROS_INFO("Not for multi robot");
	}
	SimpleLaserScan *node = new SimpleLaserScan(n, 36, robot_name);
	node->run(-1);
	return EXIT_SUCCESS;
}
