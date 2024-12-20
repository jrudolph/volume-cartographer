#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "z5/factory.hxx"
#include <nlohmann/json.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>

namespace fs = std::filesystem;

using json = nlohmann::json;

static inline cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return{a[0]*b[0],a[1]*b[1]};
}

static float dot_s(const cv::Vec3f &p)
{
    return p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
}

template <typename E>
float ldist(const E &p, const cv::Vec3f &tgt_o, const cv::Vec3f &tgt_v)
{
    return cv::norm((p-tgt_o).cross(p-tgt_o-tgt_v))/cv::norm(tgt_v);
}

template <typename E>
static float search_min_line(const cv::Mat_<E> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt_o, cv::Vec3f tgt_v, cv::Vec2f init_step, float min_step_x)
{
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains({loc[0],loc[1]})) {
        out = {-1,-1,-1};
        return -1;
    }
    
    bool changed = true;
    E val = at_int(points, loc);
    out = val;
    float best = ldist(val, tgt_o, tgt_v);
    float res;
    
    //TODO check maybe add more search patterns, compare motion estimatino for video compression, x264/x265, ...
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    cv::Vec2f step = init_step;
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+mul(off,step);
            
            //just skip if out of bounds
            if (!boundary.contains({cand[0],cand[1]}))
                continue;
                
                val = at_int(points, cand);
                res = ldist(val, tgt_o, tgt_v);
                if (res < best) {
                    changed = true;
                    best = res;
                    loc = cand;
                    out = val;
                }
        }
        
        if (changed)
            continue;
        
        step *= 0.5;
        changed = true;
        
        if (step[0] < min_step_x)
            break;
    }
    
    return best;
}

template <typename E>
float line_off(const E &p, const cv::Vec3f &tgt_o, const cv::Vec3f &tgt_v)
{
    return (tgt_o-p).dot(tgt_v)/dot_s(tgt_v);
}

using IntersectVec = std::vector<std::pair<float,cv::Vec2f>>;

float surf_th = 0.5;

IntersectVec getIntersects(const cv::Vec2i &seed, const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &step)
{
    cv::Vec3f o = points(seed[1],seed[0]);
    cv::Vec3f n = grid_normal(points, {seed[0],seed[1],seed[2]});
    if (std::isnan(n[0]))
        return {};
    std::vector<cv::Vec2f> locs = {seed};
    uint32_t sr = seed[1];
    for(int i=0;i<1000;i++)
    {
        cv::Vec2f loc = {rand_r(&sr) % points.cols, seed[1] - 50 + (rand_r(&sr) % 100)};
        cv::Vec3f res;
        float dist = search_min_line(points, loc, res, o, n, step, 0.01);
        
        if (dist > 0.5 || dist < 0)
            continue;
        
        if (!loc_valid_xy(points,loc))
            continue;
        
        // std::cout << dist << res << loc << std::endl;
        
        bool found = false;
        for(auto l : locs) {
            if (cv::norm(loc, l) <= 4) {
                found = true;
                break;
            }
        }
        if (!found)
            locs.push_back(loc);
    }

    IntersectVec dist_locs;
    for(auto l : locs)
        dist_locs.push_back({line_off(at_int(points,l),o,n), l});
    
    //just sort by normal position and let median take care of the rest
    // std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.first < b.first; });
    // return dist_locs;    
    
    std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.second[0] > b.second[0]; });
    
    //we could have two groups (other part of the scroll), in that case the x locations should be between the ones of the first group!

    // for(auto p : dist_locs)
        // std::cout << p.first << p.second << std::endl;
    
    bool two_halves = false;
    for(int i=1;i<dist_locs.size()-1;i++) 
        if (abs(dist_locs[i-1].first - dist_locs[i+1].first) < std::min(abs(dist_locs[i-1].first - dist_locs[i].first),abs(dist_locs[i].first - dist_locs[i+1].first))) {
            two_halves = true;
        }
        
    // std::cout << "two " << two_halves << std::endl;
    
    if (!two_halves) {
        std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.second[0] < b.second[0]; });
        // std::cout << std::endl;
        return dist_locs;
    }

    IntersectVec a, b;
    std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.first < b.first; });
    a.push_back(dist_locs[0]);
    b.push_back(dist_locs.back());
    dist_locs.erase(dist_locs.begin());
    dist_locs.erase(dist_locs.begin()+dist_locs.size()-1);
    
    bool seed_in_a = (a.back().first == 0.01);
    
    for(auto pair : dist_locs) {
        // std::cout << pair.first << std::endl;
        if (abs(pair.first - a.back().first) < abs(pair.first - b.back().first)) {
            a.push_back(pair);
            if (pair.first == 0.01)
                seed_in_a = true;
        }
        else
            b.push_back(pair);
    }
    
    // std::cout << "seed " << seed_in_a << std::endl;
//     
    if (seed_in_a) {
        dist_locs = a;
    }
    else
        dist_locs = b;
    
    std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.second[0] < b.second[0]; });
    // std::cout << "out" << std::endl;
    // for(auto p : dist_locs)
        // std::cout << p.first << p.second << std::endl;
        
    // std::cout << std::endl;
    return dist_locs;
}

//l is [y, x]!
bool loc_valid_nan(const cv::Mat_<float> &m, const cv::Vec2d &l)
{
    if (std::isnan(l[0]))
        return false;
    
    cv::Rect bounds = {0, 0, m.rows-2,m.cols-2};
    cv::Vec2i li = {floor(l[0]),floor(l[1])};
    
    if (!bounds.contains(li))
        return false;
    
    if (std::isnan(m(li[0],li[1])))
        return false;
    if (std::isnan(m(li[0]+1,li[1])))
        return false;
    if (std::isnan(m(li[0],li[1]+1)))
        return false;
    if (std::isnan(m(li[0]+1,li[1]+1)))
        return false;
    return true;
}

bool loc_valid_nan_xy(const cv::Mat_<float> &m, const cv::Vec2d &l)
{
    return loc_valid_nan(m, {l[1],l[0]});
}

float find_wind_x(cv::Mat_<float> &winding, cv::Vec2f &loc, float tgt_wind)
{
    if (!loc_valid_nan_xy(winding, loc))
        return -1;
    
    float best_diff = abs(at_int(winding,loc)-tgt_wind);
    
    std::vector<cv::Vec2f> neighs = {{1,0},{-1,0}};
    
    float step = 16.0;
    bool updated = true;
    while (updated)
    {
        updated = false;
        for (auto n : neighs) {
            cv::Vec2f cand = loc + step*n;
            if (!loc_valid_nan_xy(winding, cand))
                continue;
            float diff = abs(at_int(winding,cand)-tgt_wind);
            if (diff < best_diff) {
                best_diff = diff;
                loc = cand;
                updated = true;
                break;
            }
        }
        
        if (!updated) {
            if (step <= 0.001)
                break;
            step /= 2;
            updated = true;
        }
    }

    return best_diff;
}

std::string time_str()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);
    
    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y%m%d%H%M%S");
    oss << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}

int main(int argc, char *argv[])
{
    if (argc != 3 && argc != 7) {
        std::cout << "usage: " << argv[0] << " <tiffxyz> <out>" << std::endl;
        std::cout << "or: " << argv[0] << " <tiffxyz> <out> <cropx> <cropy> <cropyw> <croph>" << std::endl;
        return EXIT_SUCCESS;
    }
    
    fs::path seg_path = argv[1];
    fs::path obj_path = argv[2];
    
    QuadSurface *surf = nullptr;
    try {
        surf = load_quad_from_tifxyz(seg_path);
    }
    catch (...) {
        std::cout << "error when loading: " << seg_path << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat_<cv::Vec3f> points = surf->rawPoints();
    
    if (argc == 7)
        points = points(cv::Rect(atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),atoi(argv[6])));
    
    cv::Mat_<cv::Vec3b> img(points.size(), 0);
    
    // cv::Vec2i seed = {1145, 168};
    
    int num_conn = 500;
    
    std::vector<IntersectVec> intersects(num_conn);
    
#pragma omp parallel
    {
        unsigned int sr = omp_get_thread_num();
#pragma omp for
        for(int i=0;i<num_conn;i++) {
            
            
            cv::Vec2i seed = {rand_r(&sr) % points.cols, rand_r(&sr) % points.rows};
            while (points(seed[1],seed[0])[0] == -1)
                seed = {rand_r(&sr) % points.cols, rand_r(&sr) % points.rows};

            intersects[i] = getIntersects(seed, points, surf->_scale);
        }
    }
    
    // for(auto pair : intersects) {
    //     std::cout << pair.first << pair.second << std::endl;
    // }
    
    for(auto &iv : intersects) {
        cv::Vec3b col = {50+rand() % 155,50+rand() % 155,50+rand() % 155};
        for(auto &pair : iv) {
            // img(pair.second[1],pair.second[0]) = col;
            // std::cout << pair.first << pair.second << std::endl;
            cv::circle(img, cv::Point(pair.second), 3, col, -1);
        }
        // std::cout << std::endl;
    }
    cv::imwrite("dbg.tif", img);
    
    std::vector<std::vector<int>> wind_dists_x(points.cols/100+1);
    
    std::cout << "x size " << wind_dists_x.size() << std::endl;
    
    for(auto &iv : intersects) {
        if (iv.size() > 4)
            for(int n=0;n<iv.size()-1;n++) {
                int x1 = iv[n].second[0];
                int x2 = iv[n+1].second[0];
                float dist = x2-x1;
                // std::cout << dist << std::endl;
                wind_dists_x[x1/100].push_back(dist);
                wind_dists_x[x2/100].push_back(dist);
            }
    }
    
    for(auto &dists : wind_dists_x)
        std::sort(dists.begin(), dists.end());
    
    std::vector<int> wind_x_ref(wind_dists_x.size());
    
    std::cout << "final out " << wind_dists_x.size() << std::endl;
    int last = 0;
    int first = 0;
    for(int i=0;i<wind_dists_x.size();i++) {
        std::vector<int> &dists = wind_dists_x[i];

        if (dists.size()) {
            last = dists[dists.size()/2];
            if (!first) {
                first = last;
                for (int j=0;j<i;j++)
                    wind_x_ref[j] = first;
            }
        }
        wind_x_ref[i] = last;
    }
    
    for(auto w : wind_x_ref)
        std::cout << "wind x step: " << w << std::endl;
    // std::cout << dists[dists.size()/2] << std::endl;
    
    cv::Mat_<float> winding(points.size(), 0);
    cv::Mat_<float> wind_w(points.size(), 0);
    
    cv::Vec2i seed = {intersects[0][0].second[1],intersects[0][0].second[0]};
    
    std::vector<cv::Vec3b> wind_cols;
    for(int i=0;i<400;i++) {
        cv::Vec3b col = {50+rand() % 127,50+rand() % 127,50+rand() % 127};
        col[rand()%3] = 192+rand()%63;
        if (i%2 == 0)
            col *= 0.5;
        wind_cols.push_back(col);
    }
    
    int ramp = 50;
    bool expanded = true;
    int it = 0;
    while (expanded || ramp >= -50) {
        winding(seed) = 0;
        wind_w(seed) = 1;
        
        if (!expanded)
            ramp--;
        
        expanded = false;
        
        it++;
        
        cv::Mat_<float> winding_out = winding.clone();
        cv::Mat_<float> wind_w_out = wind_w.clone();
        
        float rough_w = std::max(0.0, ramp/50.0);
        
        for(auto &iv : intersects) {
            //FIXME make it go both ways!
            if (iv.size() < 4)
                continue;
            for(int n=0;n<iv.size()-1;n++) {
                int x1 = iv[n].second[0];
                int x2 = iv[n+1].second[0];
                
                cv::Vec2i p1i = {iv[n].second[1],iv[n].second[0]};
                cv::Vec2i p2i = {iv[n+1].second[1],iv[n+1].second[0]};
                
                int ref_x = wind_x_ref[(x1+x2)/100];
                
                // std::cout << abs(x2-x1 - ref_x) << " " << x2-x1 << " vs " << ref_x << " wot " << x1 << " " << x2 << p1i << p2i << std::endl;
                
                if (abs(x2-x1 - ref_x) > ref_x/3)
                    continue;
                
                if (wind_w(p1i) == 0 && wind_w(p2i) == 0) {
                    // std::cout << "both 0" << std::endl;
                    continue;
                }
                
                // std::cout << "go" << std::endl;
                
                if (wind_w(p2i) == 0) {
                    expanded = true;
                    wind_w(p2i) = wind_w(p1i);
                    winding(p2i) = winding(p1i)+1;
                    wind_w_out(p2i) = wind_w(p1i);
                    winding_out(p2i) = winding(p1i)+1;
                }
                else if (wind_w(p1i) == 0) {
                    expanded = true;
                    wind_w(p1i) = wind_w(p2i);
                    winding(p1i) = winding(p2i)-1;
                    wind_w_out(p1i) = wind_w(p2i);
                    winding_out(p1i) = winding(p2i)-1;
                }
                else {
                    float avg_wind = (wind_w(p1i)*winding(p1i) + wind_w(p2i)*(winding(p2i)-1))/(wind_w(p1i)+wind_w(p2i));
                    winding_out(p1i) = avg_wind;
                    winding_out(p2i) = avg_wind+1;
                    float avg_w = (wind_w(p1i)*wind_w(p1i) + wind_w(p2i)*wind_w(p2i))/(wind_w(p1i)+wind_w(p2i));
                    wind_w_out(p1i) = avg_w;
                    wind_w_out(p2i) = avg_w;
                }
            }
            // break;
        }
        
        if (rough_w == 1.0) {
            winding_out.copyTo(winding);
            wind_w_out.copyTo(wind_w);
        }
        else {
            // std::cout << "w " << rough_w << std::endl;
            winding = winding_out*rough_w + winding*(1-rough_w);
            wind_w = wind_w_out*rough_w + wind_w*(1-rough_w);
        }
        
        cv::Rect bounds_inv(0,0,points.rows-1,points.cols-1);
        
        std::vector<cv::Vec2i> neighs = {{0,-1},{0,1},{1,0},{-1,0},{1,1},{-1,1},{1,-1},{-1,-1},{0,-4},{0,4},{4,0},{-4,0},{0,-16},{0,16},{16,0},{-16,0}};
#pragma omp parallel for
        for(int j=1;j<winding.rows-1;j++)
            for(int i=1;i<winding.cols-1;i++) {
                cv::Vec2i p = {j,i};
                
                if (points(p)[0] == -1)
                    continue;
                
                float w = wind_w(p);
                float sum = winding(p)*w;
                for(auto n : neighs) {
                    cv::Vec2i pn = p + n;
                    
                    if (!bounds_inv.contains(pn))
                        continue;
                    
                    if (points(pn)[0] == -1)
                        continue;
                    
                    int dist = std::max(abs(n[1]),abs(n[0]));
                    if (dist > 1)
                    {
                        bool skip = false;
                        cv::Vec2i step = n/dist;
                        for(int s=1;s<dist;s++)
                            if (points(p+step*s)[0] == -1) {
                                skip = true;
                                break;
                            }
                        if (skip)
                            continue;
                    }
                    
                    float mul = 1.0;
                    if (rough_w != 1.0 && dist > 1) {
                        mul = rough_w/dist;
                    }
                    
                    sum += (winding(pn)-float(n[1])/wind_x_ref[pn[1]/100])*wind_w(pn)*mul;
                    w += wind_w(pn)*mul;
                }
                if (w != 0) {
                    if (wind_w_out(p) == 0.0)
                        expanded = true;
                    winding_out(p) = sum/w;
                    wind_w_out(p) = 1;
                }
            }
            
        winding_out.copyTo(winding);
        wind_w_out.copyTo(wind_w);
            
        if (it % 50 == 0 || ramp == -50) {
            cv::imwrite("wind_w.tif", wind_w);
            std::cout << "finished it " << it << std::endl;
            
            cv::Mat_<cv::Vec3b> vis(points.size(), {0,0,0});
            cv::Mat_<float> winding_err(points.size());
#pragma omp parallel for
            for(int j=0;j<winding.rows;j++)
                for(int i=0;i<winding.cols;i++) {
                    if (wind_w(j,i)) {
                        int w_num = std::min(std::max(int(winding(j,i)*2+200),0),398);
                        float f = winding(j,i)*2+100 - int(winding(j,i)*2+100);
                        vis(j,i) = wind_cols[w_num]*(1-f)+wind_cols[w_num+1]*f;
                    }
                }
#pragma omp parallel for
            for(int j=0;j<winding.rows;j++)
                for(int i=0;i<winding.cols-1;i++) {
                    if (wind_w(j,i) && wind_w(j,i+1)) {
                        float wind_g = winding(j,i+1) - winding(j,i);
                        wind_g = wind_g * wind_x_ref[i/100];
                        winding_err(j,i) = abs(1-wind_g);
                    }
                }
            // cv::imwrite("winding"+std::to_string(n)+".tif", winding);
            // cv::imwrite("wind_vis"+std::to_string(n)+".tif", vis);
            cv::imwrite("winding.tif", winding);
            cv::imwrite("wind_vis.tif", vis);
            cv::imwrite("wind_err.tif", winding_err);
        }
        
        if (ramp == -50)
            break;
    }
    
    for(int j=0;j<winding.rows;j++)
        for(int i=0;i<winding.cols;i++)
            if (wind_w(j,i) == 0)
                winding(j,i) = NAN;

    cv::Mat_<cv::Vec3f> out(points.size(), {-1,-1,-1});
    double curr_wind = -2;
    for(int i=0;i<winding.cols-1;i++) {
        // std::cout << "final out gen " << i << std::endl;
#pragma omp parallel for
        for(int j=0;j<winding.rows;j++) {
            float diff = -1;
            cv::Vec2f loc(i,j);
            bool succ = false;
            for(int r=0;r<100;r++) {
                diff = find_wind_x(winding, loc, curr_wind);
                if (diff >= 0 && diff <= 0.1) {
                    succ = true;
                    break;
                }
                loc = {rand() % points.cols,j};
            }
            
            if (succ) {
                // std::cout << loc << cv::Vec2i(i, j) << points.size() << std::endl;
                out(j,i) = at_int(points, loc);
            }
        }
        //FIXME inaccurate, also do median by winding number?
        curr_wind += 1.0/wind_x_ref[i/100];
    }
    
    std::vector<cv::Mat> chs;
    cv::split(out, chs);
    
    cv::imwrite("newx.tif",chs[0]);
    
    QuadSurface *surf_full = new QuadSurface(out, surf->_scale);
    
    fs::path tgt_dir = "/home/hendrik/data/ml_datasets/vesuvius/manual_wget/dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/";
    std::string name_prefix = "testing_fill_";
    std::string uuid = name_prefix + time_str();
    fs::path seg_dir = tgt_dir / uuid;
    std::cout << "saving " << seg_dir << std::endl;
    surf_full->save(seg_dir, uuid);
    
    return EXIT_SUCCESS;
}
