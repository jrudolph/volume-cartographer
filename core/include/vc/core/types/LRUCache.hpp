#pragma once

#include <list>
#include <unordered_map>

namespace volcart
{
/**
 * @class LRUCache
 * @brief Least Recently Used Cache
 * @author Sean Karlage
 *
 * A cache using a least recently used replacement policy. As elements are used,
 * they are moved to the front of the cache. When the cache exceeds capacity,
 * elements are popped from the end of the cache and replacement elements are
 * added to the front.
 *
 * Usage information is tracked in a std::unordered_map since its performance
 * will likely be the fastest of the STL classes. This should be profiled. Data
 * elements are stored in insertion order in a std::list and are pointed to
 * by the elements in the usage map.
 *
 * Design mostly taken from
 * <a href = "https://github.com/lamerman/cpp-lru-cache">here</a>.
 *
 * @ingroup Types
 */
template <typename TKey, typename TValue>
class LRUCache
{
public:
    /**
     * @brief Templated Key/Value pair
     *
     * Stored in the data list.
     */
    using TPair = typename std::pair<TKey, TValue>;

    /**
     * @brief Templated Key/Value pair iterator
     *
     * Stored in the LRU map.
     */
    using TListIterator = typename std::list<TPair>::iterator;

    /**@{*/
    /** @brief Default constructor */
    LRUCache() : capacity_{DEFAULT_CAPACITY} {}

    /** @brief Constructor with cache capacity parameter */
    explicit LRUCache(size_t capacity) : capacity_{capacity} {}
    /**@}*/

    /**@{*/
    /** @brief Set the maximum number of elements in the cache */
    void setCapacity(size_t newCapacity)
    {
        if (newCapacity <= 0) {
            throw std::invalid_argument(
                "Cannot create cache with capacity <= 0");
        } else {
            capacity_ = newCapacity;
        }

        // Cleanup elements that exceed the capacity
        while (lookup_.size() > capacity_) {
            auto last = std::end(items_);
            last--;
            lookup_.erase(last->first);
            items_.pop_back();
        }
    }

    /** @brief Get the maximum number of elements in the cache */
    size_t capacity() const { return capacity_; }

    /** @brief Get the current number of elements in the cache */
    size_t size() const { return lookup_.size(); }
    /**@}*/

    /**@{*/
    /**
     * @brief Get an item from the cache by key
     *
     * Returns a const reference because we typically use the LRUCache for
     * cv::Mat data. Returning a const ref is better because then if you try to
     * modify the value without explicitly calling .clone() you'll get a
     * compile error.
     */
    const TValue& get(const TKey& k)
    {
        auto lookupIter = lookup_.find(k);
        if (lookupIter == std::end(lookup_)) {
            throw std::invalid_argument("Key not in cache");
        } else {
            items_.splice(std::begin(items_), items_, lookupIter->second);
            return lookupIter->second->second;
        }
    }

    /** @brief Put an item into the cache */
    void put(const TKey& k, const TValue& v)
    {
        // If already in cache, need to refresh it
        auto lookupIter = lookup_.find(k);
        if (lookupIter != std::end(lookup_)) {
            items_.erase(lookupIter->second);
            lookup_.erase(lookupIter);
        }

        items_.push_front(TPair(k, v));
        lookup_[k] = std::begin(items_);

        if (lookup_.size() > capacity_) {
            auto last = std::end(items_);
            last--;
            lookup_.erase(last->first);
            items_.pop_back();
        }
    }

    /** @brief Check if an item is already in the cache */
    bool exists(const TKey& k) { return lookup_.find(k) != std::end(lookup_); }

    /** @brief Clear the cache */
    void purge()
    {
        lookup_.clear();
        items_.clear();
    }
    /**@}*/

private:
    /** Default cache capacity */
    static constexpr size_t DEFAULT_CAPACITY = 200;
    /** Cache data storage */
    std::list<TPair> items_;
    /** Cache usage information */
    std::unordered_map<TKey, TListIterator> lookup_;
    /** Maximum number of elements in the cache */
    size_t capacity_;
};
}  // namespace volcart