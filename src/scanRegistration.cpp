// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;

const int systemDelay = 0;                      // 系统延时参数
int systemInitCount = 0;                        // 系统初始化次数
bool systemInited = false;                      // 系统初始化标志位
int N_SCANS = 0;                                // 雷达线束数量
float cloudCurvature[400000];                   // 曲率，每次扫描最多40ｗ个点
int cloudSortInd[400000];                       // 曲率索引
int cloudNeighborPicked[400000];                // 选中标志位
int cloudLabel[400000];                         // 数据点类别

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;

double MINIMUM_RANGE = 0.1;                      // 去除近处点云参数阈值

/*
* @brief: 去除近处的点云
*
* @param[in]: 　 cloud_in         输入点云
* @param[out]: 　cloud_out        去除近处点的点云
* @param:        thres            去除近处点参数阈值
*/

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    // 如果点云不是同一个，需要同步点云时间戳并resize点云大小
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    // 更新过滤后点云的数量
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    // 这几个变量手册上说是Mandatory，必须要设定的
    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

/**
 * @brief      激光点云回调处理函数
 *
 * @param     sensor_msgs::PointCloud2ConstPtr     laser cloud message
 * 
 * 
 *
 * 
 */

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    // 系统延时：通过控制订阅到的雷达点云帧数，控制点云处理开始时刻
    // systemDelay参数默认为０
    if (!systemInited)
    { 
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    TicToc t_whole;
    TicToc t_prepare;
    // 
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    // ros消息转成pcl点云
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;

    // pcl去除无效点云，indices保留的是有效点索引
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    // 去除近处点
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);


    int cloudSize = laserCloudIn.points.size();

    // 这里要搞清楚为什么要加负号，需要明白velodyne的坐标系以及旋转方向
    // velodyne坐标系为：ｘ轴朝向前，y轴朝向左，z轴朝向上，右手系
    // 由于右手系，因此雷达顺时针旋转，角度应该为负
    // atan2的范围[-pi, pi]
    // 结束角度加2pi，使得整个旋转周期为２pi,变为[pi, 3pi]
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;

    // 控制整个点云角度在[pi, 3pi]
    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    bool halfPassed = false;
    int count = cloudSize;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        // 计算scan俯仰角，确定scanID
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;

        if (N_SCANS == 16)
        {
            // +-15°的垂直视场，垂直角度分辨率2°，-15°时的scanID = 0
            scanID = int((angle + 15) / 2 + 0.5);
            // scanID[0,15]
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            // 垂直角度分辨率1.33°，-30.67°时的scanID = 0, 垂直视场范围[-30.67, 10.56]
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            // 垂直角度分辨率0.4254°，+2°时的scanID = 0,-8.83°时scanID = 32
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5); // scanID[0,32]
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            // 过滤大于２°，小于-24.33°的角度数据，其实scanID[0,50]已经满足要求
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);

        float ori = -atan2(point.y, point.x);
        if (!halfPassed)
        { 
            // 确保-pi/2 < (ori - startOri) < 3*pi/2
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }
            // 旋转角度超过起始角度180°（过半）
            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            // 确保 -3*pi/2 < (ori - endOri) < pi/2
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }

        // 雷达点相对该帧数据起始点的角度占整个周期角度的百分比
        float relTime = (ori - startOri) / (endOri - startOri);
        // 整数部分是scanID，小数部分代表该激光点距sweep开始时刻的时间差
        point.intensity = scanID + scanPeriod * relTime;
        // 该帧scan的所有雷达点
        laserCloudScans[scanID].push_back(point); 
    }
    
    cloudSize = count;
    printf("points size %d \n", cloudSize);

    // 将多帧scan按顺序拼接成一帧点云
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++)
    { 
        // scanStartInd[i]和scanEndInd[i]保证每帧scan收尾去除5个点，用于计算边缘点和平面点
        scanStartInd[i] = laserCloud->size() + 5;
        // 存储还是保存了所有点
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;
    }

    printf("prepare time %f \n", t_prepare.toc());

    // 使用每个点的前后5个点计算曲率
    for (int i = 5; i < cloudSize - 5; i++)
    { 
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
        // 曲率
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        // 曲率点索引
        cloudSortInd[i] = i;
        // 初始化都未被选中
        cloudNeighborPicked[i] = 0;
        // 初始都标记为surfPointsLessFlat
        cloudLabel[i] = 0;
    }


    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;       // 曲率很大的边线点云
    pcl::PointCloud<PointType> cornerPointsLessSharp;   // 曲率较大的边线点云（包含曲率很大的边线点云）
    pcl::PointCloud<PointType> surfPointsFlat;          // 很平的平面点云(点云数量不多)
    pcl::PointCloud<PointType> surfPointsLessFlat;      // 较平的平面点云(包含很平的平面点云，需要降采样)

    float t_q_sort = 0;
    for (int i = 0; i < N_SCANS; i++)
    {
        // 筛除该帧scan不足6个点的点云
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;        
       
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);

        // 将每帧scan均分为6段
        for (int j = 0; j < 6; j++)
        {
            // 每一段的起点索引
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            // 每一段的终点索引
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;
            // 每一段按照曲率升序排序
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            t_q_sort += t_tmp.toc();

            // 曲率从大到小,提取边线点
            int largestPickedNum = 0;            
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k]; // 这一段曲率点索引

                // 如果最大曲率点曲率大于一定阈值，且未被选中过
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {
                    // 最大曲率点选中的数量
                    largestPickedNum++;
                    
                    // 选择曲率最大的2个点
                    if (largestPickedNum <= 2)
                    {                        
                        cloudLabel[ind] = 2;    // 类别标记为1,表示曲率很大的边线点
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    // 选择曲率较大的18个点
                    else if (largestPickedNum <= 20)
                    {                        
                        cloudLabel[ind] = 1;    // 类别标记为1, 表示曲率较大的边线点
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;  // 选取超过20个点后，该段不再选取边线点
                    }

                    // 标记该点已被选中
                    cloudNeighborPicked[ind] = 1; 

                    // 选中的点前后各５个点中，标记附近点，保证提取特征点相对分散
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1; // 选中边线点附近点也标记上，保证后面不再提取
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 曲率从小到大，提取平面点
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];  // 曲率点的索引

                // 如果曲率点的曲率小于一定阈值，并且未被选中过
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {

                    cloudLabel[ind] = -1;   // 标记为-1,表示曲率很小的平面点
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    // 小曲率平面点选中数量
                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    { 
                        break;  // 平面点选中超过4个，不再提取平面点
                    }
                    // 标记为选中
                    cloudNeighborPicked[ind] = 1;

                    // 选中的点前后各５个点中，标记附近点，保证提取特征点相对分散
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1; // 选中平面点附近点也标记上，保证后面不再提取
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 将cloudLabel小于等于0的点，都当做不那么平的平面点
            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }

        // 平面点云降采样
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    // 注意发布的点云消息，时间戳均以当前雷达点云消息时间戳

    // 发布当前拼接起来的点云
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    // 发布曲率很大的边线点云
    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    // 发布曲率较大的边线点云
    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    // 发布曲率很小的平面点云
    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    // 发布曲率较小的平面点云（包含曲率很小的平面点云且经过降采样）
    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    // pub each scan
    // 是否发布每一帧scan
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;

    nh.param<int>("scan_line", N_SCANS, 16);

    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    // 订阅原始雷达点云消息并处理
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    // 发布拼接起来的点云
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    // 发布曲率很大的边线点云
    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    // 发布曲率较大的边线点云
    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    // 发布曲率很小的平面点云
    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    // 发布曲率较小的平面点云
    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    // 发送每一帧scan
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();

    return 0;
}
