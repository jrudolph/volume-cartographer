#pragma once

#include "vc/core/types/PerPixelMap.hpp"
#include "vc/core/types/Texture.hpp"
#include "vc/core/types/Volume.hpp"

namespace volcart
{
namespace texturing
{
/**
 * @class IntegralTexture
 * @author Seth Parker
 * @date 11/24/2016
 *
 * @brief Generate a Texture by taking the discrete integral (summation) of the
 * linear neighborhood adjacent to a point
 *
 * @ingroup Texture
 */
class IntegralTexture
{
public:
    /**
     * @brief Weight option
     *
     * Setting the weight option applies a linear weight factor to the intensity
     * values of the neighborhood. The options are named according to which
     * values along a point's surface normal are favored.
     *
     * The weight factors for the options are as follows:
     * - Positive: Most Positive: 1.0, Least Positive: 0.0
     * - Negative: Most Positive: 0.0, Least Positive: 1.0
     * - None: Most Positive: 1.0, Least Positive: 1.0
     */
    enum class Weight { Positive, Negative, None };

    /**@{*/
    /** @brief Set the input PerPixelMap */
    void setPerPixelMap(PerPixelMap ppm) { ppm_ = std::move(ppm); }

    /** @brief Set the input Volume */
    void setVolume(Volume::Pointer vol) { vol_ = std::move(vol); }

    /** @brief Set the sampling search radius: the distance from the mesh to
     * consider for compositing */
    void setSamplingRadius(double r) { radius_ = r; }

    /**
     * @brief Set the sampling interval: how frequently the voxels along the
     * radius are sampled for compositing purposes
     *
     * Default = 1.0
     */
    void setSamplingInterval(double i) { interval_ = i; }

    /**
     * @brief Set the filtering search direction: which "side" of the mesh to
     * consider when compositing
     *
     * Default: Bidirectional
     */
    void setSamplingDirection(Direction d) { direction_ = d; }

    /**
     * @brief Set the weight option
     *
     * Default: None
     */
    void setWeight(Weight w) { weightType_ = w; }
    /**@}*/

    /**@{*/
    /** @brief Compute the Texture */
    Texture compute();
    /**@}*/

    /**@{*/
    /** @brief Get the generated Texture */
    const Texture& getTexture() const { return result_; }

    /** @copydoc getTexture() const */
    Texture& getTexture() { return result_; }
    /**@}*/

private:
    /** PPM */
    PerPixelMap ppm_;
    /** Volume */
    Volume::Pointer vol_;
    /** Search radius */
    double radius_;
    /** Search direction */
    Direction direction_{Direction::Bidirectional};
    /** Search sampling interval */
    double interval_{1.0};
    /** Result */
    Texture result_;
    /** Weighting option */
    Weight weightType_{Weight::None};
    /** Current weight value */
    double currentWeight_;
    /** Current weight increment value */
    double weightStep_;
    /** Setup the weight values */
    void setup_weights_(size_t s);
};

}  // texturing
}  // volcart
