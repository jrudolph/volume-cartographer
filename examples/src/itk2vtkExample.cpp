//
// Created by Media Team on 6/24/15.
//


/*
 * Purpose: Run volcart::meshing::itk2vtk() and write results to file.
 *          Run volcart::meshing::vtk2itk() and write results to file.
 *          Saved file wills be read in by the itk2vtkTest.cpp file under
 *          v-c/testing/meshing.
 *
 *          Need to use out plywriter because the vtkPLYwriter does not
 *          include normals when it writes to file
 */

#include "common/vc_defines.h"
#include "common/shapes/Plane.h"
#include "common/shapes/Cube.h"
#include "common/shapes/Arch.h"
#include "common/shapes/Sphere.h"
#include "common/shapes/Cone.h"
#include "meshing/itk2vtk.h"
#include <vtkPLYWriter.h>
#include <common/io/objWriter.h>

void writePLYHeaderAndPoints(std::ostream &out, vtkPolyData* mesh);

int main( int argc, char* argv[] ) {

  //init shapes --> used by both conversion directions
  volcart::shapes::Plane Plane;
  volcart::shapes::Cube Cube;
  volcart::shapes::Arch Arch;
  volcart::shapes::Sphere Sphere;
  volcart::shapes::Cone Cone;

  //
  //vtk2itk conversions
  //

  //init VC_MeshType::Pointer objects to hold the output of vtk2itk conversions
  VC_MeshType::Pointer out_ITKPlane = VC_MeshType::New();
  VC_MeshType::Pointer out_ITKCube = VC_MeshType::New();
  VC_MeshType::Pointer out_ITKArch = VC_MeshType::New();
  VC_MeshType::Pointer out_ITKSphere = VC_MeshType::New();
  VC_MeshType::Pointer out_ITKCone = VC_MeshType::New();

  //vtk2itk() calls
  volcart::meshing::vtk2itk(Plane.vtkMesh(), out_ITKPlane);
  volcart::meshing::vtk2itk(Cube.vtkMesh(), out_ITKCube);
  volcart::meshing::vtk2itk(Arch.vtkMesh(), out_ITKArch);
  volcart::meshing::vtk2itk(Sphere.vtkMesh(), out_ITKSphere);
  volcart::meshing::vtk2itk(Cone.vtkMesh(), out_ITKCone);


  //
  // write itk meshes to file
  //

  volcart::io::objWriter writer;

  //cycle through the shapes and write
  int ShapeCounter = 0;
  while (ShapeCounter < 5){

    if (ShapeCounter == 0){

      //write itk meshes
      writer.setMesh(out_ITKPlane);
      writer.setPath("VTKPlaneMeshConvertedToITK.obj");
      writer.write();
    }
    else if (ShapeCounter == 1){

      writer.setMesh(out_ITKCube);
      writer.setPath("VTKCubeMeshConvertedToITK.obj");
      writer.write();
    }
    else if (ShapeCounter == 2){

      writer.setMesh(out_ITKArch);
      writer.setPath("VTKArchMeshConvertedToITK.obj");
      writer.write();
    }
    else if (ShapeCounter == 3){

      writer.setMesh(out_ITKSphere);
      writer.setPath("VTKSphereMeshConvertedToITK.obj");
      writer.write();
    }
    else if (ShapeCounter == 4){

      writer.setMesh(out_ITKCone);
      writer.setPath("VTKConeMeshConvertedToITK.obj");
      writer.write();
    }

      ++ShapeCounter;
  }

  /* write vtk meshes to file
   *
   * This is a terrible implementation...but was having issues with scope, so writing explicitly in
   * for each shape to complete the examples for itk2vtkTest
   *
   */

  std::ofstream MeshOutputFileStream;
  int NumberOfVTKPoints, NumberOfVTKCells;

  //Prepare to start writing loop
  int VTKShapeCounter = 0;

  while (VTKShapeCounter < 5){

    //plane
    if (VTKShapeCounter == 0) {

      vtkPolyData *out_VTKPlaneMesh = vtkPolyData::New();
      volcart::meshing::itk2vtk(Plane.itkMesh(), out_VTKPlaneMesh);
      NumberOfVTKPoints = out_VTKPlaneMesh->GetNumberOfPoints();
      NumberOfVTKCells = out_VTKPlaneMesh->GetNumberOfCells();
      MeshOutputFileStream.open("ITKPlaneMeshConvertedToVTK.ply");

      // write header
      MeshOutputFileStream << "ply" << std::endl
      << "format ascii 1.0" << std::endl
      << "comment Created by particle simulation https://github.com/viscenter/registration-toolkit" << std::endl
      << "element vertex " << NumberOfVTKPoints << std::endl
      << "property float x" << std::endl
      << "property float y" << std::endl
      << "property float z" << std::endl
      << "property float nx" << std::endl
      << "property float ny" << std::endl
      << "property float nz" << std::endl
      << "element face " << NumberOfVTKCells << std::endl
      << "property list uchar int vertex_indices" << std::endl
      << "end_header" << std::endl;

      for (int point = 0; point < NumberOfVTKPoints; point++) {

        // x y z
        MeshOutputFileStream << out_VTKPlaneMesh->GetPoint(point)[0] << " "
        << out_VTKPlaneMesh->GetPoint(point)[1] << " "
        << out_VTKPlaneMesh->GetPoint(point)[2] << " ";

        MeshOutputFileStream << out_VTKPlaneMesh->GetPointData()->GetNormals()->GetTuple(point)[0] << " "
        << out_VTKPlaneMesh->GetPointData()->GetNormals()->GetTuple(point)[1] << " "
        << out_VTKPlaneMesh->GetPointData()->GetNormals()->GetTuple(point)[2] << std::endl;
      }

      for ( vtkIdType c_id = 0; c_id < NumberOfVTKCells; c_id++){

        vtkCell *out_VTKCell = out_VTKPlaneMesh->GetCell(c_id);

        MeshOutputFileStream << "3 " <<  out_VTKCell->GetPointIds()->GetId(0) << " "
        << out_VTKCell->GetPointIds()->GetId(1) << " "
        << out_VTKCell->GetPointIds()->GetId(2) << std::endl;
      }

    }
      // cube
    else if (VTKShapeCounter == 1){

      vtkPolyData *out_VTKCubeMesh = vtkPolyData::New();
      volcart::meshing::itk2vtk(Cube.itkMesh(), out_VTKCubeMesh);
      NumberOfVTKPoints = out_VTKCubeMesh->GetNumberOfPoints();
      NumberOfVTKCells = out_VTKCubeMesh->GetNumberOfCells();
      MeshOutputFileStream.open("ITKCubeMeshConvertedToVTK.ply");

      // write header
      MeshOutputFileStream << "ply" << std::endl
      << "format ascii 1.0" << std::endl
      << "comment Created by particle simulation https://github.com/viscenter/registration-toolkit" << std::endl
      << "element vertex " << NumberOfVTKPoints << std::endl
      << "property float x" << std::endl
      << "property float y" << std::endl
      << "property float z" << std::endl
      << "property float nx" << std::endl
      << "property float ny" << std::endl
      << "property float nz" << std::endl
      << "element face " << NumberOfVTKCells << std::endl
      << "property list uchar int vertex_indices" << std::endl
      << "end_header" << std::endl;

      for (int point = 0; point < NumberOfVTKPoints; point++) {

        // x y z
        MeshOutputFileStream << out_VTKCubeMesh->GetPoint(point)[0] << " "
        << out_VTKCubeMesh->GetPoint(point)[1] << " "
        << out_VTKCubeMesh->GetPoint(point)[2] << " ";

        MeshOutputFileStream << out_VTKCubeMesh->GetPointData()->GetNormals()->GetTuple(point)[0] << " "
        << out_VTKCubeMesh->GetPointData()->GetNormals()->GetTuple(point)[1] << " "
        << out_VTKCubeMesh->GetPointData()->GetNormals()->GetTuple(point)[2] << std::endl;
      }

      for ( vtkIdType c_id = 0; c_id < NumberOfVTKCells; c_id++){

        vtkCell *out_VTKCell = out_VTKCubeMesh->GetCell(c_id);

        MeshOutputFileStream << "3 " << out_VTKCell->GetPointIds()->GetId(0) << " "
        << out_VTKCell->GetPointIds()->GetId(1) << " "
        << out_VTKCell->GetPointIds()->GetId(2) << std::endl;
      }

    }
      // arch
    else if (VTKShapeCounter == 2){

      vtkPolyData *out_VTKArchMesh = vtkPolyData::New();
      volcart::meshing::itk2vtk(Arch.itkMesh(), out_VTKArchMesh);
      NumberOfVTKPoints = out_VTKArchMesh->GetNumberOfPoints();
      NumberOfVTKCells = out_VTKArchMesh->GetNumberOfCells();
      MeshOutputFileStream.open("ITKArchMeshConvertedToVTK.ply");

      // write header
      MeshOutputFileStream << "ply" << std::endl
      << "format ascii 1.0" << std::endl
      << "comment Created by particle simulation https://github.com/viscenter/registration-toolkit" << std::endl
      << "element vertex " << NumberOfVTKPoints << std::endl
      << "property float x" << std::endl
      << "property float y" << std::endl
      << "property float z" << std::endl
      << "property float nx" << std::endl
      << "property float ny" << std::endl
      << "property float nz" << std::endl
      << "element face " << NumberOfVTKCells << std::endl
      << "property list uchar int vertex_indices" << std::endl
      << "end_header" << std::endl;

      for (int point = 0; point < NumberOfVTKPoints; point++) {

        // x y z
        MeshOutputFileStream << out_VTKArchMesh->GetPoint(point)[0] << " "
        << out_VTKArchMesh->GetPoint(point)[1] << " "
        << out_VTKArchMesh->GetPoint(point)[2] << " ";

        MeshOutputFileStream << out_VTKArchMesh->GetPointData()->GetNormals()->GetTuple(point)[0] << " "
        << out_VTKArchMesh->GetPointData()->GetNormals()->GetTuple(point)[1] << " "
        << out_VTKArchMesh->GetPointData()->GetNormals()->GetTuple(point)[2] << std::endl;
      }

      for ( vtkIdType c_id = 0; c_id < NumberOfVTKCells; c_id++){

        vtkCell *out_VTKCell = out_VTKArchMesh->GetCell(c_id);

        MeshOutputFileStream << "3 " << out_VTKCell->GetPointIds()->GetId(0) << " "
        << out_VTKCell->GetPointIds()->GetId(1) << " "
        << out_VTKCell->GetPointIds()->GetId(2) << std::endl;
      }

    }
      // sphere
    else if (VTKShapeCounter == 3){

      vtkPolyData *out_VTKSphereMesh = vtkPolyData::New();
      volcart::meshing::itk2vtk(Sphere.itkMesh(), out_VTKSphereMesh);
      NumberOfVTKPoints = out_VTKSphereMesh->GetNumberOfPoints();
      NumberOfVTKCells = out_VTKSphereMesh->GetNumberOfCells();
      MeshOutputFileStream.open("ITKSphereMeshConvertedToVTK.ply");
      // write header
      MeshOutputFileStream << "ply" << std::endl
      << "format ascii 1.0" << std::endl
      << "comment Created by particle simulation https://github.com/viscenter/registration-toolkit" << std::endl
      << "element vertex " << NumberOfVTKPoints << std::endl
      << "property float x" << std::endl
      << "property float y" << std::endl
      << "property float z" << std::endl
      << "property float nx" << std::endl
      << "property float ny" << std::endl
      << "property float nz" << std::endl
      << "element face " << NumberOfVTKCells << std::endl
      << "property list uchar int vertex_indices" << std::endl
      << "end_header" << std::endl;

      for (int point = 0; point < NumberOfVTKPoints; point++) {

        // x y z
        MeshOutputFileStream << out_VTKSphereMesh->GetPoint(point)[0] << " "
        << out_VTKSphereMesh->GetPoint(point)[1] << " "
        << out_VTKSphereMesh->GetPoint(point)[2] << " ";

        MeshOutputFileStream << out_VTKSphereMesh->GetPointData()->GetNormals()->GetTuple(point)[0] << " "
        << out_VTKSphereMesh->GetPointData()->GetNormals()->GetTuple(point)[1] << " "
        << out_VTKSphereMesh->GetPointData()->GetNormals()->GetTuple(point)[2] << std::endl;
      }

      for ( vtkIdType c_id = 0; c_id < NumberOfVTKCells; c_id++){

        vtkCell *out_VTKCell = out_VTKSphereMesh->GetCell(c_id);

        MeshOutputFileStream << "3 " << out_VTKCell->GetPointIds()->GetId(0) << " "
        << out_VTKCell->GetPointIds()->GetId(1) << " "
        << out_VTKCell->GetPointIds()->GetId(2) << std::endl;
      }

    }
      // cone
    else if (VTKShapeCounter == 4){

      vtkPolyData *out_VTKConeMesh = vtkPolyData::New();
      volcart::meshing::itk2vtk(Cone.itkMesh(), out_VTKConeMesh);
      NumberOfVTKPoints = out_VTKConeMesh->GetNumberOfPoints();
      NumberOfVTKCells = out_VTKConeMesh->GetNumberOfCells();
      MeshOutputFileStream.open("ITKConeMeshConvertedToVTK.ply");

      // write header
      MeshOutputFileStream << "ply" << std::endl
      << "format ascii 1.0" << std::endl
      << "comment Created by particle simulation https://github.com/viscenter/registration-toolkit" << std::endl
      << "element vertex " << NumberOfVTKPoints << std::endl
      << "property float x" << std::endl
      << "property float y" << std::endl
      << "property float z" << std::endl
      << "property float nx" << std::endl
      << "property float ny" << std::endl
      << "property float nz" << std::endl
      << "element face " << NumberOfVTKCells << std::endl
      << "property list uchar int vertex_indices" << std::endl
      << "end_header" << std::endl;

      for (int point = 0; point < NumberOfVTKPoints; point++) {

        // x y z
        MeshOutputFileStream << out_VTKConeMesh->GetPoint(point)[0] << " "
        << out_VTKConeMesh->GetPoint(point)[1] << " "
        << out_VTKConeMesh->GetPoint(point)[2] << " ";

        MeshOutputFileStream << out_VTKConeMesh->GetPointData()->GetNormals()->GetTuple(point)[0] << " "
        << out_VTKConeMesh->GetPointData()->GetNormals()->GetTuple(point)[1] << " "
        << out_VTKConeMesh->GetPointData()->GetNormals()->GetTuple(point)[2] << std::endl;
      }

      for ( vtkIdType c_id = 0; c_id < NumberOfVTKCells; c_id++){

        vtkCell *out_VTKCell = out_VTKConeMesh->GetCell(c_id);

        MeshOutputFileStream << "3 " << out_VTKCell->GetPointIds()->GetId(0) << " "
        << out_VTKCell->GetPointIds()->GetId(1) << " "
        << out_VTKCell->GetPointIds()->GetId(2) << std::endl;
      }
    }

    MeshOutputFileStream.close();

    //move to the next shape
    ++VTKShapeCounter;

  } //while


  return EXIT_SUCCESS;
}