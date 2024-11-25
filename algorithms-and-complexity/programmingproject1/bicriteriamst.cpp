#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

struct Edge {
    int u, v;          // Nodes
    int profit, weight; // Profit and weight
};

// Union-Find for Kruskal's Algorithm
class UnionFind {
private:
    vector<int> parent, rank;
public:
    UnionFind(int n) {
        parent.resize(n + 1);
        rank.resize(n + 1, 0);
        for (int i = 0; i <= n; ++i) parent[i] = i;
    }
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    bool unionSets(int x, int y) {
        int rootX = find(x), rootY = find(y);
        if (rootX == rootY) return false;
        if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else {
            parent[rootY] = rootX;
            ++rank[rootX];
        }
        return true;
    }
};

// Greedy Algorithm to Build Spanning Tree with Modified Profits
pair<long long, long long> calculateSpanningTree(int n, vector<Edge>& edges, double lambda) {
    vector<pair<double, int>> modifiedEdges;
    for (int i = 0; i < edges.size(); ++i) {
        double modifiedProfit = edges[i].profit - lambda * edges[i].weight;
        modifiedEdges.emplace_back(modifiedProfit, i);
    }

    // Sort edges by descending modified profit
    sort(modifiedEdges.rbegin(), modifiedEdges.rend());

    UnionFind uf(n);
    long long totalProfit = 0, totalWeight = 0;
    int edgesUsed = 0;

    // Replace structured bindings with explicit access to pair members
    for (size_t i = 0; i < modifiedEdges.size(); ++i) {
        double modProfit = modifiedEdges[i].first;
        int idx = modifiedEdges[i].second;
        const Edge& e = edges[idx];
        if (uf.unionSets(e.u, e.v)) {
            totalProfit += e.profit;
            totalWeight += e.weight;
            edgesUsed++;
            if (edgesUsed == n - 1) break; // Stop when spanning tree is formed
        }
    }

    return make_pair(totalProfit, totalWeight);
}

// Binary Search to Maximize the Profit-to-Weight Ratio
pair<long long, long long> findOptimalTree(int n, vector<Edge>& edges) {
    double low = 0.0, high = 1e9; // Set bounds for binary search
    const double EPS = 1e-9;     // Precision threshold
    pair<long long, long long> result;

    while (high - low > EPS) {
        double mid = (low + high) / 2.0;
        auto tree = calculateSpanningTree(n, edges, mid);
        long long p = tree.first, w = tree.second;

        if (w == 0) { // Avoid division by zero
            high = mid;
            continue;
        }

        double ratio = static_cast<double>(p) / w;
        if (ratio > mid) {
            result = tree;
            low = mid; // Increase lower bound
        } else {
            high = mid; // Decrease upper bound
        }
    }

    return result;
}

// Utility Function to Compute GCD
long long gcd(long long a, long long b) {
    return b == 0 ? a : gcd(b, a % b);
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<Edge> edges(m);

    for (int i = 0; i < m; ++i) {
        cin >> edges[i].u >> edges[i].v >> edges[i].profit >> edges[i].weight;
    }

    // Find the optimal tree
    pair<long long, long long> result = findOptimalTree(n, edges);
    long long pTotal = result.first, wTotal = result.second;

    // Reduce by GCD
    long long divisor = gcd(pTotal, wTotal);
    cout << (pTotal / divisor) << " " << (wTotal / divisor) << endl;

    return 0;
}
