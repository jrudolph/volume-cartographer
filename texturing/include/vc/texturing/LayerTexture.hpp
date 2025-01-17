#pragma once

/** @file */

#include "vc/texturing/TexturingAlgorithm.hpp"

#include "vc/core/neighborhood/LineGenerator.hpp"

namespace volcart::texturing
{
/**
 * @class LayerTexture
 * @author Seth Parker
 * @date 11/24/2016
 *
 * @brief Generate a Texture of layered images
 *
 * The Texture generated by this class contains multiple texture images. Each
 * image is the projection of the mesh some distance through the Volume along
 * each point's surface normal. For well-formed meshes and parameterizations,
 * this amounts to resampling the Volume into a flattened subvolume with the
 * segmentation mesh forming a straight line at its center.
 *
 * @ingroup Texture
 */
class LayerTexture : public TexturingAlgorithm
{
public:
    /** Pointer type */
    using Pointer = std::shared_ptr<LayerTexture>;

    /** Make shared pointer */
    static auto New() -> Pointer { return std::make_shared<LayerTexture>(); }

    /** Default destructor */
    ~LayerTexture() override = default;

    /**
     * @brief Set the Neighborhood generator
     *
     * This class only supports LineGenerator
     */
    void setGenerator(LineGenerator::Pointer g) { gen_ = std::move(g); }

    /**@{*/
    /** @brief Compute the Texture */
    auto compute() -> Texture override;
    /**@}*/
private:
    /** Neighborhood Generator */
    LineGenerator::Pointer gen_;
};

}  // namespace volcart::texturing
