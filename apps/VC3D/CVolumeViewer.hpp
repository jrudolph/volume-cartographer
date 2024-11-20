// CVolumeViewer.h
// Chao Du 2015 April
#pragma once

#include <QtWidgets>
#include <opencv2/core/core.hpp>

#include <set>

class ChunkCache;
class Surface;
class SurfacePointer;

class QGraphicsScene;

namespace volcart {
    class Volume;
}

namespace ChaoVis
{

class CVolumeViewerView;
class CSurfaceCollection;
class POI;
class Intersection;

class CVolumeViewer : public QWidget
{
    Q_OBJECT

public:
    CVolumeViewer(CSurfaceCollection *col, QWidget* parent = 0);
    ~CVolumeViewer(void);

    void setCache(ChunkCache *cache);
    void setSurface(const std::string &name);
    void renderVisible(bool force = false);
    void renderIntersections();
    cv::Mat render_area(const cv::Rect &roi);
    void invalidateVis();
    void invalidateIntersect(const std::string &name = "");
    
    std::set<std::string> intersects();
    void setIntersects(const std::set<std::string> &set);
    std::string surfName() { return _surf_name; };
    void recalcScales();
    void renderPoints();
    
    CVolumeViewerView* fGraphicsView;

public slots:
    void OnVolumeChanged(std::shared_ptr<volcart::Volume> vol);
    void onVolumeClicked(QPointF scene_loc,Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onPanRelease(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onPanStart(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onSurfaceChanged(std::string name, Surface *surf);
    void onPOIChanged(std::string name, POI *poi);
    void onIntersectionChanged(std::string a, std::string b, Intersection *intersection);
    void onScrolled();
    void onZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers modifiers);
    void onCursorMove(QPointF);
    void onPointsChanged(const std::vector<cv::Vec3f> red, const std::vector<cv::Vec3f> blue);

signals:
    void SendSignalSliceShift(int shift, int axis);
    void SendSignalStatusMessageAvailable(QString text, int timeout);
    void sendVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void sendShiftNormal(cv::Vec3f step);

protected:
    void ScaleImage(double nFactor);
    void CenterOn(const QPointF& point);

protected:
    // widget components
    QGraphicsScene* fScene;

    // data
    QImage* fImgQImage;
    bool fSkipImageFormatConv;

    QGraphicsPixmapItem* fBaseImageItem;
    
    std::shared_ptr<volcart::Volume> volume = nullptr;
    Surface *_surf = nullptr;
    SurfacePointer *_ptr = nullptr;
    cv::Vec2f _vis_center = {0,0};
    std::string _surf_name;
    int axis = 0;
    int loc[3] = {0,0,0};
    
    ChunkCache *cache = nullptr;
    QRect curr_img_area = {0,0,1000,1000};
    float _scale = 0.5;
    float _scene_scale = 1.0;
    float _ds_scale = 0.5;
    int _ds_sd_idx = 1;
    float _max_scale = 1;
    float _min_scale = 1;

    QLabel *_lbl = nullptr;

    float _z_off = 0.0;
    
    QGraphicsItem *_center_marker = nullptr;
    QGraphicsItem *_cursor = nullptr;
    
    bool _slice_vis_valid = false;
    std::vector<QGraphicsItem*> slice_vis_items; 
    
    std::set<std::string> _intersect_tgts = {"visible_segmentation"};
    std::unordered_map<std::string,std::vector<QGraphicsItem*>> _intersect_items;
    Intersection *_ignore_intersect_change = nullptr;
    
    CSurfaceCollection *_surf_col = nullptr;
    
    std::vector<cv::Vec3f> _red_points;
    std::vector<cv::Vec3f> _blue_points;
    std::vector<QGraphicsItem*> _points_items; 
};  // class CVolumeViewer

}  // namespace ChaoVis
