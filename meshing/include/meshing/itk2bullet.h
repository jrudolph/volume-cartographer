#pragma once

#ifdef VC_USE_BULLET

#include <BulletSoftBody/btSoftBody.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>

#include "core/vc_defines.h"

namespace volcart
{
namespace meshing
{
/**
 * @class itk2bullet
 * @author Abigail Coleman
 * @date 10/21/15
 *
 * @brief Convert from an ITKMesh to a btSoftBody.
 *
 * Copy vertex and face information from a btSoftBody to an ITKMesh.
 *
 * @ingroup Meshing
 */
class itk2bullet
{
public:
    /**
    * @param worldInfo Soft Body world configuration
    */
    itk2bullet(
        ITKMesh::Pointer input,
        btSoftBodyWorldInfo& worldInfo,
        btSoftBody** output);
};

/**
 * @class bullet2itk
 * @author Abigail Coleman
 * @date 10/21/15
 *
 * @brief Convert from a btSoftBody to an ITKMesh.
 *
 * Copy vertex and face information from a btSoftBody to an ITKMesh.
 *
 * @ingroup Meshing
 */
class bullet2itk
{
public:
    bullet2itk(btSoftBody* input, ITKMesh::Pointer output);
};

}  // namespace meshing
}  // namespace volcart

#endif  // VC_USE_BULLET
