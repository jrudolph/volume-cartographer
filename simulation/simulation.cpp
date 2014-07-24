#include <iostream>

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#define WINDOW "select"
#define REFERENCE "/Users/mike/Desktop/italia/003.tif"

#define FIELDSIZE 400
#define ITERATION 1000

#define SCALE 10

#define SPRING_CONSTANT_K -0.1
#define SPRING_RESTING_X 1

typedef cv::Vec3f Particle;
typedef Particle Force;

Force*** field;
std::vector<Particle> particle;
std::vector<cv::Point> click_list;

// based on the interpolation formula from
// http://paulbourke.net/miscellaneous/interpolation/
Force interpolate_field(Particle point) {
  double dx, dy, dz, int_part;
  dx = modf(point(0), &int_part);
  dy = modf(point(1), &int_part);
  dz = modf(point(2), &int_part);

  int x_min, x_max, y_min, y_max, z_min, z_max;
  x_min = (int)point(0);
  x_max = x_min + 1;
  y_min = (int)point(1);
  y_max = y_min + 1;
  z_min = (int)point(2);
  z_max = z_min + 1;

  Force vector =
    field[x_min][y_min][z_min] * (1 - dx) * (1 - dy) * (1 - dz) +
    field[x_max][y_min][z_min] * dx       * (1 - dy) * (1 - dz) +
    field[x_min][y_max][z_min] * (1 - dx) * dy       * (1 - dz) +
    field[x_min][y_min][z_max] * (1 - dx) * (1 - dy) * dz +
    field[x_max][y_min][z_max] * dx       * (1 - dy) * dz +
    field[x_min][y_max][z_max] * (1 - dx) * dy       * dz +
    field[x_max][y_max][z_min] * dx       * dy       * (1 - dz) +
    field[x_max][y_max][z_max] * dx       * dy       * dz;

  return vector;
}

Force spring_force(int index) {
  Force f(0,0,0);
  if (index != particle.size() - 1) {
    Particle to_right = particle[index] - particle[index + 1];
    double length = sqrt(to_right.dot(to_right));
    normalize(to_right, to_right, SPRING_CONSTANT_K * (length - SPRING_RESTING_X));
    f += to_right;
  }
  if (index != 0) {
    Particle to_left = particle[index] - particle[index - 1];
    double length = sqrt(to_left.dot(to_left));
    normalize(to_left, to_left, SPRING_CONSTANT_K * (length - SPRING_RESTING_X));
    f += to_left;
  }
  return f;
}

void update_particles() {
  std::vector<Force> workspace(particle.size());
  for(int i = 0; i < particle.size(); ++i)
    workspace[i] = interpolate_field(particle[i]) + spring_force(i);
  for(int i = 0; i < particle.size(); ++i)
    particle[i] += workspace[i];
}

void add_particle(int event, int x, int y, int flags, void* userdata) {
  if  (event == cv::EVENT_LBUTTONDOWN)
    click_list.push_back(cv::Point(x, y));
}

int main(int argc, char* argv[]) {

  if (argc == 1) {
    std::cout << "FEED MEEE" << std::endl;
    exit(EXIT_FAILURE);
  }

  cv::Mat ref = cv::imread(REFERENCE);
  if(!ref.data) {
    std::cout << "Error loading reference image" << std::endl;
    return -1;
  }
  cv::namedWindow(WINDOW, 1);
  cv::setMouseCallback(WINDOW, add_particle, NULL);
  imshow(WINDOW, ref);
  cv::waitKey(0);
  for (int i = 1; i < click_list.size(); ++i) {
    cv::line(ref, click_list[i-1], click_list[i], cv::Scalar(0,255,0));
  }
  imshow(WINDOW, ref);
  cv::waitKey(0);
  cv::destroyWindow(WINDOW);

  for (int i = 1; i < click_list.size(); ++i) {
    cv::LineIterator it(ref, click_list[i-1], click_list[i]);
    while (it.pos() != click_list[i]) {
      cv::Point pos = it.pos();
      particle.push_back(Particle(pos.x, pos.y, 3));
      it++;
    }
  }


  // allocate and initalize force field
  field = new Force**[FIELDSIZE];
  for (int i = 0; i < FIELDSIZE; ++i) {
    field[i] = new Force*[FIELDSIZE];
    for (int j = 0; j < FIELDSIZE; ++j) {
      field[i][j] = new Force[FIELDSIZE];
      for (int k = 0; k < FIELDSIZE; ++k) {
        field[i][j][k] = Force(0,0,0);
      }
    }
  }

  // read points from files and add to force field
  int point_counter = 0;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  for (int i = 1; i < argc; ++i) {
    if (i && i % 10 == 0) {
      std::cout << i << " files read" << std::endl;
    }
    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal> (argv[i], *cloud) == -1) {
      PCL_ERROR ("Couldn't read file\n");
      exit(EXIT_FAILURE);
    } else {
      pcl::PointCloud<pcl::PointXYZRGBNormal>::iterator point;

      for (point = cloud->begin(); point != cloud->end(); ++point) {
        int x, y, z;
        y = point->x;
        x = point->y;
        z = point->z;

        field[x][y][z](0) = point->normal[0]/SCALE;
        field[x][y][z](1) = point->normal[1]/SCALE;
        field[x][y][z](2) = point->normal[2]/SCALE;

        point_counter++;

      }
    }
  }

  // particles can be started anywhere
  // z=2 is the first index with nonzero
  // force field vectors in my test data

  // for (int i = 0; i < FIELDSIZE - 1; ++i) {
  //   Particle p(i, 50, 2);
  //   particle.push_back(p);
  // }

  std::cout << std::endl << "running particle simulation" << std::endl;

  // run particle simulation
  for (int step = 0; step < ITERATION; ++step) {

    if (step && step % 10 == 0) {
      std::cout << "step " << step << " completed" << std::endl;
    }

    std::ofstream csv;
    csv.open((std::string)"particle.csv." + std::to_string(step));
    csv << "x,y,z" << std::endl;

    for (int i = 0; i < particle.size(); ++i) {
      csv << particle[i](0) << "," << particle[i](1) << "," << particle[i](2) << std::endl;
    }

    update_particles();

    csv.close();
  }

  exit(EXIT_SUCCESS);
}