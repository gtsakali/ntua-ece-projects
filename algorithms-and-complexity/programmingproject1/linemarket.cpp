#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// Function to determine if a given distance is feasible
bool isFeasible(const vector<pair<int, int>>& intervals, int itemCount, int minDistance) {
    int placedItems = 0;
    int lastPosition = -1e9; // Start with a very small value

    for (const auto& range : intervals) {
        int currentPos = max(range.first, lastPosition + minDistance); // Earliest valid position in the range
        while (currentPos <= range.second) {
            ++placedItems;
            lastPosition = currentPos;
            if (placedItems == itemCount) return true;
            currentPos += minDistance; // Move to the next potential position
        }
    }
    return false;
}

int main() {
    int itemCount, intervalCount;
    cin >> itemCount >> intervalCount;

    vector<pair<int, int>> intervals(intervalCount);
    for (int i = 0; i < intervalCount; ++i) {
        cin >> intervals[i].first >> intervals[i].second;
    }

    // Sort intervals by their starting points
    sort(intervals.begin(), intervals.end());

    // Binary search to find the maximum possible minimum distance
    int left = 1, right = intervals.back().second - intervals.front().first, maxDistance = 0;
    while (left <= right) {
        int midDistance = left + (right - left) / 2;
        if (isFeasible(intervals, itemCount, midDistance)) {
            maxDistance = midDistance; // Update the best result
            left = midDistance + 1;    // Look for a larger feasible distance
        } else {
            right = midDistance - 1;   // Try smaller distances
        }
    }

    cout << maxDistance << endl;
    return 0;
}
