#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>


void showHelp (char * program_name)
{
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " cloud_filename.[pcd|ply]" << std::endl;
    std::cout << "-h: SHow this help." << std::endl;
}


int main (int argc, char** argv)
{
    // show help
    if (pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help"))
    {
        showHelp (argv[0]);
        return 0;
    }

    // fetch point cloud filename in arguments | work with PCD and PLY files
    std::vector<int> filenames;
    bool file_is_pcd = false;
    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".ply");
    if (filenames.size() != 1)
    {
        filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
        if (filenames.size() != 1)
        {
            showHelp (argv[0]);
            return -1;
        }
        else
        {
            file_is_pcd = true;
        }
    }

    // load file | works with PCD and PLY files
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ>());
    if (file_is_pcd)
    {
        if (pcl::io::loadPCDFile (argv[filenames[0]], *source_cloud) < 0)
        {
            std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
            showHelp (argv[0]);
            return -1;
        }
    }
    else
    {
        if (pcl::io::loadPLYFile(argv[filenames[0]], *source_cloud) < 0)
        {
            std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
            showHelp (argv[0]);
            return -1;    
        }
    }

    // MATRIX TRANSFORMATION
    // METHOD 1: Matrix4f
    Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
    // define a rotation matrix
    float theta = M_PI/4;  // the angle of rotation in radians
    transform_1 (0, 0) = cos (theta);
    transform_1 (0, 1) = -sin (theta);
    transform_1 (1, 0) = sin (theta);
    transform_1 (1, 1) = cos (theta);
    // define a translation of 2.5 meters on the x axis
    transform_1 (0, 3) = 2.5;
    printf("Method #1: using a Matrix4f\n");
    std::cout << transform_1 << std::endl;

    // METHOD2: Affine3f
    Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
    transform_2.translation() << 2.5, 0.0, 0.0;
    transform_2.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));
    printf("Method #2: using an Affine3f\n");
    std::cout << transform_2.matrix() << std::endl;

    // executing the transformation
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    // apply transform_1 or transform_2
    pcl::transformPointCloud (*source_cloud, *transformed_cloud, transform_2);

    // visualization
    printf("\nPoint cloud colors: white = original point cloud\n \
                                  red = transformed point cloud\n");
    pcl::visualization::PCLVisualizer viewer ("Matrix transformation example");
    // define RGB colors for the point cloud
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (source_cloud, 255, 255, 255, 255);
    // viewer.addPointCloud (source_cloud, source_cloud_color_handler, "original_cloud");
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (transformed_cloud, 230, 20, 20);
    // viewer.addPointCloud (transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");

    // viewer.addCoordinateSystem (1.0, "cloud", 0);
    // viewer.setBackgroundColor (0.05, 0.05, 0.05, 0);  // setting background to a dark grey
    // viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
    // viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
    
    // display the visualizer window position
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }

    return 0;
}


