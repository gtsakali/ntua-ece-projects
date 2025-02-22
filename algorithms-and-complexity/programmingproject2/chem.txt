#include <iostream>
#include <vector>
#include <climits>
using namespace std;

// Define a constant for infinity
const long long MAX_COST = LLONG_MAX;

// Function to calculate the pairwise interaction cost for a segment [start, end]
long long calculateSegmentCost(const vector<vector<int>>& interactionMatrix, int start, int end) {
    long long totalCost = 0;
    for (int i = start; i <= end; ++i) {
        for (int j = i + 1; j <= end; ++j) {
            totalCost += interactionMatrix[i][j];
        }
    }
    return totalCost;
}

int main() {
    // Read number of items (N) and segments (K)
    int numItems, numSegments;
    cin >> numItems >> numSegments;

    // Read the pairwise interaction matrix
    vector<vector<int>> interactionMatrix(numItems + 1, vector<int>(numItems + 1, 0));
    for (int i = 1; i < numItems; ++i) {
        for (int j = i + 1; j <= numItems; ++j) {
            cin >> interactionMatrix[i][j];
        }
    }

    // Precompute segment costs for all subarrays [start, end]
    vector<vector<long long>> segmentCosts(numItems + 1, vector<long long>(numItems + 1, 0));
    for (int start = 1; start <= numItems; ++start) {
        for (int end = start; end <= numItems; ++end) {
            segmentCosts[start][end] = calculateSegmentCost(interactionMatrix, start, end);
        }
    }

    // Initialize DP table to track minimum cost for dividing items into segments
    vector<vector<long long>> dp(numSegments + 1, vector<long long>(numItems + 1, MAX_COST));
    dp[0][0] = 0; // Base case: No items and no segments result in zero cost

    // Fill the DP table
    for (int segment = 1; segment <= numSegments; ++segment) {
        for (int end = 1; end <= numItems; ++end) {
            for (int start = 0; start < end; ++start) {
                if (dp[segment - 1][start] != MAX_COST) {
                    dp[segment][end] = min(dp[segment][end], dp[segment - 1][start] + segmentCosts[start + 1][end]);
                }
            }
        }
    }

    // Output the minimum cost for dividing all items into the given number of segments
    if (dp[numSegments][numItems] == MAX_COST) {
        cout << -1 << endl; // Output -1 if it's not possible to divide items
    } else {
        cout << dp[numSegments][numItems] << endl; // Output the minimum cost
    }

    return 0;
}