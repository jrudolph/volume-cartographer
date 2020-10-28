#include "vc/segmentation/ThinnedFloodFillSegmentation.hpp"

#include <iomanip>
#include <queue>
#include <unordered_set>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/types/Color.hpp"
#include "vc/core/util/ImageConversion.hpp"
#include "vc/core/util/Iteration.hpp"
#include "vc/core/util/Logging.hpp"

namespace fs = boost::filesystem;

using namespace volcart;
using namespace volcart::segmentation;

using TFF = ThinnedFloodFillSegmentation;
using Voxel = cv::Vec3i;

struct VoxelHash {
    size_t operator()(const Voxel& v) const
    {
        // Hash from:
        // https://dmauro.com/post/77011214305/a-hashing-function-for-x-y-z-coordinates
        auto max = std::max({v[0], v[1], v[2]});
        size_t hash = (max * max * max) + (2 * max * v[2]) + v[2];
        if (max == v[2]) {
            auto val = std::max({v[0], v[1]});
            hash += val * val;
        }
        if (v[1] >= v[0]) {
            hash += v[0] + v[1];
        } else {
            hash += v[1];
        }
        return hash;
    }
};

using VoxelList = std::vector<cv::Vec3i>;
using VoxelSet = std::unordered_set<Voxel, VoxelHash>;

struct VoxelPair {
    VoxelPair() = default;
    VoxelPair(const Voxel& v, const Voxel& parent) : v{v}, parent{parent} {}

    Voxel v;
    Voxel parent;
};

static VoxelList GetNeighbors(const Voxel& v)
{
    return {{v[0] - 1, v[1] - 1, v[2]}, {v[0], v[1] - 1, v[2]},
            {v[0] + 1, v[1] - 1, v[2]}, {v[0] - 1, v[1], v[2]},
            {v[0] + 1, v[1], v[2]},     {v[0] - 1, v[1] + 1, v[2]},
            {v[0], v[1] + 1, v[2]},     {v[0] + 1, v[1] + 1, v[2]}};
}

static VoxelSet FindIntersections(const VoxelSet& pts)
{
    VoxelSet intersections;
    for (const auto& v : pts) {
        int branchCtr = 0;
        for (const auto& n : GetNeighbors(v)) {
            if (pts.find(n) != pts.end()) {
                branchCtr++;
            }
        }
        if (branchCtr > 2) {
            // Add the voxel as an 'intersection point' that has at least one
            // branch to prune.
            intersections.insert(v);
        }
    }
    return intersections;
}

/*
 * Search the skeleton for spurs.
 * An 'intersection' where more than two paths are available contains a spur.
 * */
static VoxelSet PruneSpurs(VoxelSet skeleton, size_t spurLength)
{
    auto intersections = FindIntersections(skeleton);

    // Prune the shortest branch of all intersections (alternatively, use a
    // user-defined threshold to determine if a spur should be pruned.)
    VoxelSet visited;
    for (const auto& intPt : intersections) {
        // For each intersection, check all directly adjacent points that are
        // part of the skeleton. Search the paths connected to all adjacent
        // points (independently of each other) to find the size of
        // the possible spur (If pruning by user-defined threshold, BFS does not
        // have to complete if the size of the segment is > the user-defined
        // threshold. Don't prune in that case.)
        // If not pruning by user-defined threshold, remove the shortest
        // branch (probably the spur.) Make sure the intersection point is only
        // connected to 2 other points now.
        for (const auto& n : GetNeighbors(intPt)) {
            // Skip this neighbor if it's not in the skeleton
            if (skeleton.find(n) == skeleton.end()) {
                continue;
            }

            // Begin BFS from this point, moving away from the intersection
            // point along the branch.
            std::queue<Voxel> q;
            // Visited list ensures we don't travel in the wrong direction:
            visited.clear();

            // Queue the voxel and mark parent and voxel as visited
            q.push(n);
            visited.insert(intPt);
            visited.insert(n);

            while (!q.empty()) {
                // pick a voxel off the queue
                auto vox = q.front();
                q.pop();

                // check neighbors; if they're in the skeleton, add to queue
                for (const auto& neighbor : GetNeighbors(vox)) {
                    // If the point is in the skeleton and the point
                    // has not already been visited, add it to the
                    // queue:
                    auto inSkeleton = skeleton.find(neighbor) != skeleton.end();
                    auto inVisited = visited.find(neighbor) != visited.end();
                    if (inSkeleton and not inVisited) {
                        q.push(neighbor);
                        visited.insert(neighbor);
                    }
                }
            }

            // Measure the 'length' or number of points that we can get
            // to via this branch. We can get this from the visited
            // vector, just account for the parent being in the visited
            // vector.
            auto length = visited.size() - 1;
            // TODO: for now, the user needs to provide a fixed number.
            // TODO: later, try just pruning the shortest of the branches.
            if (length > 1 && length <= spurLength) {
                logger->debug("Removing a {}-voxel spur.", length);
                for (const Voxel& v : visited) {
                    skeleton.erase(v);
                }
            }
        }
        // TODO: prune the shortest of the branches. Do only 2 branches remain?
        // If not, repeat...
    }
    return skeleton;
}

static bool ThinPts(int dir, VoxelSet& pts)
{
    std::vector<Voxel> ptsToRemove;
    for (const Voxel& v : pts) {
        int x = v[0];
        int y = v[1];
        int z = v[2];

        bool a1 = pts.find({x, y + 1, z}) != pts.end();
        bool a2 = pts.find({x + 1, y + 1, z}) != pts.end();
        bool a3 = pts.find({x + 1, y, z}) != pts.end();
        bool a4 = pts.find({x + 1, y - 1, z}) != pts.end();
        bool a5 = pts.find({x, y - 1, z}) != pts.end();
        bool a6 = pts.find({x - 1, y - 1, z}) != pts.end();
        bool a7 = pts.find({x - 1, y, z}) != pts.end();
        bool a8 = pts.find({x - 1, y + 1, z}) != pts.end();

        // Calculate 'chi', the crossing number.
        int chi = (a1 != a3) + (a3 != a5) + (a5 != a7) + int(a7 != a1) +
                  (2 * (a2 > a1) && (a2 > a3)) + ((a4 > a3) && (a4 > a5)) +
                  ((a6 > a5) && (a6 > a7)) + ((a8 > a7) && (a8 > a1));

        // Obtain sigma -- a count of the number of 8-connected neighbors of
        // this pixel that are also in the mask.
        int sigma = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;

        // Skip this unless chi == 2 and sigma != 1
        if (chi != 2 || sigma == 1) {
            continue;
        }

        // Directional thinning
        Voxel shouldntFind(x, y, z);
        Voxel shouldFind(x, y, z);

        // "North" points
        if (dir == 0) {
            shouldntFind[1] += 1;
            shouldFind[1] -= 1;
        }
        // "South" points
        else if (dir == 1) {
            shouldntFind[1] -= 1;
            shouldFind[1] += 1;
        }
        // "East" points
        else if (dir == 2) {
            shouldntFind[0] += 1;
            shouldFind[0] -= 1;
        }
        // "West" points
        else if (dir == 3) {
            shouldntFind[0] -= 1;
            shouldFind[0] += 1;
        }

        // Remove this point if my shouldntFind neighbor IS NOT in the points
        // and my shouldFind neighbor IS in the points
        if (pts.find(shouldntFind) == pts.end() &&
            pts.find(shouldFind) != pts.end()) {
            ptsToRemove.emplace_back(v);
        }
    }

    // Remove all points marked for removal
    for (const Voxel& v : ptsToRemove) {
        pts.erase(v);
    }

    // Report the number of removed points
    logger->debug("Removed {} points in this pass.", ptsToRemove.size());

    return !ptsToRemove.empty();
}

/*
 * Skeletonize the mask by thinning.
 * This simple algorithm is described in section 8.6.2 of "Computer Vision", 5th
 * Edition, by E.R. Davies. This thinning algorithm produces a centered,
 * continuous skeleton. (So long as the mask it is thinning is continuous.)
 * */
static VoxelSet ThinMask(VoxelSet& pts)
{
    bool nThinned{true};
    bool sThinned{true};
    bool eThinned{true};
    bool wThinned{true};
    while (nThinned || sThinned || eThinned || wThinned) {
        nThinned = ThinPts(0, pts);
        sThinned = ThinPts(1, pts);
        eThinned = ThinPts(2, pts);
        wThinned = ThinPts(3, pts);
    }
    return pts;
}

inline int EuclideanDistance(const Voxel& start, const Voxel& end)
{
    return static_cast<int>(cv::norm(end - start));
}

static VoxelList DoFloodFill(
    const VoxelSet& pts, int bound, cv::Mat img, uint16_t low, uint16_t high)
{
    std::queue<VoxelPair> q;
    VoxelList mask;
    VoxelSet visited;

    // Push all the initial points onto the queue.
    // Initial points are their own 'parents'.
    for (const auto& pt : pts) {
        auto greyVal = img.at<uint16_t>(pt[1], pt[0]);
        if (greyVal > low && greyVal < high) {
            q.emplace(pt, pt);
            visited.insert(pt);
        }
    }

    while (!q.empty()) {
        // Pick a VoxelPair off the queue
        auto pair = q.front();
        q.pop();

        //'color'/record that voxel as part of the mask
        mask.push_back(pair.v);

        // check neighbors; if they're valid according to the user-defined
        // threshold AND they are not outside the original(/parent) seed point's
        // boundary, add them to the queue
        for (const auto& neighbor : GetNeighbors(pair.v)) {
            // Make sure this voxel hasn't already been added to the
            // visited list: (We don't want to add it to the queue
            // twice...)
            if (visited.find(neighbor) != visited.end()) {
                continue;
            }

            // Make sure this neighbor is in the image bounds
            if (neighbor[0] < 0 or neighbor[0] >= img.cols or neighbor[1] < 0 or
                neighbor[1] >= img.rows) {
                continue;
            }

            // Add the valid neighbor to the queue and mark it as visited.
            auto val = img.at<uint16_t>(neighbor[1], neighbor[0]);
            auto dist = EuclideanDistance(neighbor, pair.parent);
            if (val > low && val < high && dist <= bound) {
                q.emplace(neighbor, pair.parent);
                visited.insert(neighbor);
            }
        }
    }
    return mask;
}

// Estimate thickness of page from every seed point.
// 2 basic options: vertical or horizontal. Better: estimate along
// the normal of each point in the curve. This would be especially
// helpful if this algorithm was applied to a scroll.
size_t TFF::measure_thickness_(const Voxel& seed, const cv::Mat& slice) const
{
    int xPos{seed[0]};
    int xNeg{seed[0]};
    int yPos{seed[1]};
    int yNeg{seed[1]};
    bool foundMin{false};
    bool foundMax{false};
    size_t length{1};

    while (!foundMin || !foundMax) {
        if (measureVertically_) {
            yPos++;
            yNeg--;
        } else {
            xPos++;
            xNeg--;
        }

        // We've found our bound if we're out of the image bounds now
        foundMin = xNeg < 0 or yNeg < 0;
        foundMax = xPos >= slice.cols or yPos >= slice.rows;

        // Check the negative direction
        if (!foundMin) {
            auto val = slice.at<uint16_t>(yNeg, xNeg);
            if (val < low_ or val > high_) {
                foundMin = true;
            }
        }

        // Check the positive direction
        if (!foundMax) {
            auto val = slice.at<uint16_t>(yPos, xPos);
            if (val < low_ or val > high_) {
                foundMax = true;
            }
        }

        if (!(foundMin && foundMax)) {
            length++;
        }

        // Break if our length is at the the max
        if (length == maxRadius_) {
            break;
        }
    }
    return length;
}

// Given a vector of integer measurements of page thickness, one for every
// seed point, compute the median thickness.
inline size_t ComputeMedianThickness(std::vector<size_t> meas)
{
    std::nth_element(meas.begin(), meas.begin() + meas.size() / 2, meas.end());
    return meas[meas.size() / 2];
}

static TFF::PointSet AppendVoxelSetToPointSet(
    const VoxelSet& points, TFF::PointSet result)
{
    for (const auto& v : points) {
        result.emplace_back(v[0], v[1], v[2]);
    }
    return result;
}

void TFF::setFFLowThreshold(uint16_t t) { low_ = t; }
void TFF::setFFHighThreshold(uint16_t t) { high_ = t; }
void TFF::setDistanceTransformThreshold(float t) { dtt_ = t; }
void TFF::setClosingKernelSize(int s) { kernel_ = s; }
void TFF::setMeasureVertical(bool b) { measureVertically_ = b; }
void TFF::setSpurLengthThreshold(int length) { spurLength_ = length; }
void TFF::setMaxRadius(size_t radius) { maxRadius_ = radius; }
TFF::VoxelMask TFF::getMask() const { return volMask_; }
void TFF::setDumpVis(bool b) { dumpVis_ = b; }

TFF::PointSet TFF::compute()
{
    // Setup debug vis directories
    const fs::path outputDir("debugvis");
    const fs::path maskDir(outputDir / "mask");
    const fs::path skeletonDir(outputDir / "skeleton");
    if (dumpVis_) {
        fs::create_directory(outputDir);
        fs::create_directory(maskDir);
        fs::create_directory(skeletonDir);
    }

    // Clear the outputs
    result_.clear();
    volMask_.clear();

    // Signal progress has begun
    progressStarted();

    // Initialize running points with the provided starting seeds
    // Converts double-to-int by truncation
    VoxelSet seedPoints;
    auto startSlice = std::numeric_limits<size_t>::max();
    for (const auto& pt : startingPoints_) {
        startSlice = std::min(startSlice, static_cast<size_t>(pt[2]));
        seedPoints.emplace(pt[0], pt[1], pt[2]);
    }

    // Iterate over z-slices
    for (auto it : range(iterations_)) {
        // Update progress
        progressUpdated(it);

        // Calculate the current z-index
        auto zIndex = startSlice + it;

        // Get the current (single) slice image (Of type Mat)
        auto slice = vol_->getSliceDataCopy(zIndex);

        // Estimate thickness of page from every seed point.
        std::vector<size_t> estimates;
        for (const auto& v : seedPoints) {
            estimates.emplace_back(measure_thickness_(v, slice));
        }

        // Calculate the median thickness.
        // Choose the median of the measurements to be the boundary for every
        // point.
        auto bound = ComputeMedianThickness(estimates);

        // Do flood-fill with the given seed points to the estimated thickness.
        auto sliceMask = DoFloodFill(seedPoints, bound, slice, low_, high_);

        // Save to the full volume mask
        volMask_.append(sliceMask);

        // Dump image of mask on slice
        if (dumpVis_) {
            auto i = QuantizeImage(slice, CV_8U);
            cv::cvtColor(i, i, cv::COLOR_GRAY2BGR);
            for (const Voxel& v : sliceMask) {
                i.at<cv::Vec3b>(v[1], v[0]) = color::BLUE;
            }

            std::stringstream ss;
            ss << std::setw(std::to_string(vol_->numSlices()).size())
               << std::setfill('0') << zIndex << "_mask.png";
            const auto wholeMaskPath = maskDir / ss.str();
            cv::imwrite(wholeMaskPath.string(), i);
        }

        // Convert mask to a binary image so we can apply closing and distance
        // transform operations:
        cv::Mat binaryImg = cv::Mat::zeros(slice.size(), CV_8UC1);

        // Each voxel in the mask should be assigned 'white' in the binary
        // image.
        for (const Voxel& v : sliceMask) {
            binaryImg.at<uint8_t>(v[1], v[0]) = 255;
        }

        // Apply closing to fill holes and gaps.
        cv::Mat kernel = cv::Mat::ones(kernel_, kernel_, CV_8U);
        cv::Mat closedImg;
        cv::morphologyEx(binaryImg, closedImg, cv::MORPH_CLOSE, kernel);

        // Do the distance transform.
        cv::Mat dtImg;
        cv::distanceTransform(closedImg, dtImg, cv::DIST_L2, 5);
        cv::normalize(dtImg, dtImg, 1, 0, cv::NORM_MINMAX);

        // Thin the mask slightly based on the distance transform threshold set.
        // Convert smaller mask back to a vector.
        VoxelSet dtMask;
        int imgHeight = slice.rows;
        int imgWidth = slice.cols;
        for (int j = 0; j < imgHeight; j++) {
            for (int i = 0; i < imgWidth; i++) {
                auto val = dtImg.at<float>(j, i);
                if (val > dtt_) {
                    // Add back to the mask vector all points that are greater
                    // than the set distance transform threshold.
                    dtMask.emplace(i, j, zIndex);
                }
            }
        }

        // Do the thinning algorithm
        auto thinnedMask = ThinMask(dtMask);

        // Prune spurs
        auto skeleton = PruneSpurs(thinnedMask, spurLength_);

        // Update seed points for the next iteration
        seedPoints.clear();
        for (const auto& s : skeleton) {
            seedPoints.emplace(s[0], s[1], zIndex + 1);
        }

        // Save the skeleton points to the final results
        result_ = AppendVoxelSetToPointSet(skeleton, result_);

        // Signal changes
        pointsetUpdated.send(result_);
        maskUpdated.send(volMask_);

        // Visualize the pruned skeleton if applicable
        if (dumpVis_) {
            auto i = QuantizeImage(slice, CV_8U);
            cv::cvtColor(i, i, cv::COLOR_GRAY2BGR);
            for (const Voxel& v : skeleton) {
                i.at<cv::Vec3b>(v[1], v[0]) = color::GREEN;
            }

            std::stringstream ss;
            ss << std::setw(std::to_string(vol_->numSlices()).size())
               << std::setfill('0') << zIndex << "_skeleton.png";
            const auto wholeSkeletonPath = skeletonDir / ss.str();
            cv::imwrite(wholeSkeletonPath.string(), i);
        }
    }
    progressComplete();
    return result_;
}
